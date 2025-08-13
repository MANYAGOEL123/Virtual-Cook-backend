import os
import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, LLM
import uvicorn

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

# Initialize app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini LLM setup
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=api_key,
    verbose=True,
)

# Define AI Agent
recipe_agent = Agent(
    name="Home Cook Assistant",
    role="Simple Recipe Expert",
    goal="Help everyday people cook easy meals using only the ingredients they have.",
    backstory=(
        "You are a friendly kitchen helper designed to suggest easy, step-by-step recipes "
        "using only the ingredients provided by the user. Your suggestions must be simple, "
        "require no fancy equipment, and should be understandable even to someone who has never cooked before. "
        "Avoid complex techniques, and explain everything in a very beginner-friendly tone."
    ),
    llm=llm,
    verbose=True,
)

# Request model from frontend
class IngredientRequest(BaseModel):
    # user_id: str
    ingredients: List[str]

class ChatRequest(BaseModel):
    user_id: str
    message: str

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

class MealPlanRequest(BaseModel):
    household_size: int = Field(..., gt=0, le=20)
    dietary_preference: str = "None"  # None, Vegetarian, Vegan, Keto, Gluten-free
    cuisine_preference: str = "Mixed" # Mixed, Indian, Italian, Asian, etc.
    budget_level: str = "Medium"      # Low, Medium, High
    cooking_time: str = "Standard (20–45 min)"  # Quick (≤ 20 min), Standard, Flexible
    allergies: List[str] = []
    health_goal: str = "Balanced diet"
    variety_preference: str = "High variety (no repeats)"  # or Smart reuse (reduce waste)
    reuse_ingredients: bool = True

    @validator("dietary_preference","cuisine_preference","budget_level","cooking_time","health_goal","variety_preference", pre=True)
    def strip_text(cls, v):
        return v.strip() if isinstance(v, str) else v

class DietPlanRequest(BaseModel):
    weight: float = Field(..., gt=20, lt=400)         # kg
    height: float = Field(..., gt=100, lt=250)        # cm
    age: int = Field(..., gt=10, lt=100)
    gender: str = Field(..., pattern="^(male|female)$")
    activity_level: str = Field(..., pattern="^(sedentary|light|moderate|active|very_active)$")
    target_weight: float = Field(..., gt=20, lt=400)  # kg

    # Optional preferences
    dietary_preference: str = "Any"
    cuisine_preference: str = "Any"
    allergies: List[str] = []
    cooking_time: str = "Any"        # Quick / Standard / Flexible / Any
    budget_level: str = "Any"        # Low / Medium / High / Any

# --------- Utility: Clean & Validate JSON ----------

def _clean_json_text(s: str) -> str:
    """Remove code fences and stray text around JSON."""
    txt = s.strip()
    if txt.startswith("```"):
        lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("```")]
        txt = "\n".join(lines).strip()
    # Attempt to trim to first { ... } block
    first = txt.find("{")
    last = txt.rfind("}")
    if first != -1 and last != -1 and last > first:
        txt = txt[first:last+1]
    return txt

def _ensure_7_days(data: Dict[str, Any], household_size: int) -> Dict[str, Any]:
    """Guarantee weeklyMeals has all 7 days Monday→Sunday and servings set."""
    weekly = data.get("weeklyMeals", [])
    by_day = { (item.get("day") or "").strip(): item for item in weekly if isinstance(item, dict) }

    def mk(meal_name: str):
        return {"meal": meal_name, "servings": int(household_size)}

    fixed = []
    for d in DAYS:
        item = by_day.get(d)
        if not item:
            item = {
                "day": d,
                "breakfast": mk("Simple breakfast"),
                "lunch": mk("Simple lunch"),
                "dinner": mk("Simple dinner"),
            }
        else:
            # normalize servings
            for k in ("breakfast","lunch","dinner"):
                if k not in item or not isinstance(item[k], dict):
                    item[k] = mk("Simple " + k)
                else:
                    item[k]["servings"] = int(household_size)
        fixed.append(item)

    data["weeklyMeals"] = fixed
    return data

def _categorize_grocery_list(gl) -> Dict[str, List[str]]:
    """If AI returned a list, wrap as Other; if dict, pass-through."""
    if isinstance(gl, dict):
        return gl
    if isinstance(gl, list):
        return {"Other": [str(x) for x in gl]}
    return {"Other": []}

# For diet plans
def _bmr(weight, height, age, gender):
    # Mifflin-St Jeor
    if gender == "male":
        return 10*weight + 6.25*height - 5*age + 5
    else:
        return 10*weight + 6.25*height - 5*age - 161

def _tdee(bmr, activity_level):
    mult = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
        "active": 1.725, "very_active": 1.9
    }
    return bmr * mult[activity_level]

def _ensure_7_days_diet(data: Dict[str, Any], target_cals: int) -> Dict[str, Any]:
    """Guarantee Monday→Sunday; if meals/calories missing, fill with splits."""
    weekly = data.get("weeklyPlan", [])
    by_day = { (item.get("day") or "").strip(): item for item in weekly if isinstance(item, dict) }

    def mk_split(total):
        # 25% breakfast, 35% lunch, 30% dinner, 10% snack
        return {
            "breakfast": {"meal": "Balanced breakfast", "calories": round(total*0.25)},
            "lunch":     {"meal": "Balanced lunch",     "calories": round(total*0.35)},
            "dinner":    {"meal": "Balanced dinner",    "calories": round(total*0.30)},
            "snack":     {"meal": "Light snack",        "calories": total - (round(total*0.25)+round(total*0.35)+round(total*0.30))}
        }

    fixed = []
    for d in DAYS:
        item = by_day.get(d, {"day": d})
        # Ensure each meal exists with numeric calories
        defaults = mk_split(target_cals)
        for meal in ("breakfast","lunch","dinner","snack"):
            if meal not in item or not isinstance(item[meal], dict):
                item[meal] = defaults[meal]
            else:
                # if missing fields
                item[meal]["meal"] = item[meal].get("meal") or defaults[meal]["meal"]
                cals = item[meal].get("calories")
                item[meal]["calories"] = int(cals) if isinstance(cals, (int,float)) else defaults[meal]["calories"]
        fixed.append(item)

    data["weeklyPlan"] = fixed
    # normalize dailyCalories to target
    data["dailyCalories"] = int(target_cals)
    return data


@app.post("/generate-recipe")
async def generate_recipe(request: IngredientRequest):
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="No ingredients provided")

    ingredient_list = ", ".join(request.ingredients)
    description = (
        f"Generate a creative and complete recipe using ONLY the following ingredients: "
        f"{ingredient_list}. Do not include any ingredients not in this list. "
        f"Return the recipe in the following format:\n\n"
        f"Recipe Name: <name>\nDescription: <short description>\nIngredients: <list>\nSteps: <list>"
    )

    task = Task(
        description=description,
        agent=recipe_agent,
        expected_output="A recipe strictly using only the provided ingredients.",
    )

    crew = Crew(agents=[recipe_agent], tasks=[task])

    try:
        result = await asyncio.create_task(crew.kickoff_async())
        return {"recipe": result.raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# General chat endpoint (e.g., greetings)
@app.post("/chat")
async def chat_handler(request: ChatRequest):
    message = request.message.lower()

    # If message includes ingredients or "recipe"
    keywords = ["ingredient", "have", "recipe", ","]
    if any(k in message for k in keywords):
        description = (
            f"Suggest an easy-to-understand recipe using only these ingredients: {request.message}. "
            f"Do not add any ingredients not mentioned. Keep instructions beginner-friendly."
        )
        expected = "A recipe suggestion using only the given ingredients."
    else:
        # For general messages like "hello", "who are you?", etc.
        description = (
            f"Reply like a helpful cooking assistant. If the user says hello or asks who you are, "
            f"respond by saying you're a recipe assistant who can help suggest meals using ingredients they have."
        )
        expected = "A friendly introduction message."

    task = Task(
        description=description,
        agent=recipe_agent,
        expected_output=expected,
    )

    crew = Crew(agents=[recipe_agent], tasks=[task])

    try:
        result = await asyncio.create_task(crew.kickoff_async())
        return {"response": result.raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-mealplan")
async def generate_mealplan(request: MealPlanRequest):
    # Build strict JSON prompt with schema + constraints
    description = f"""
You are an expert meal planner. Generate a weekly family meal plan for the following requirements:

- Household size: {request.household_size}
- Dietary preference: {request.dietary_preference}
- Cuisine preference: {request.cuisine_preference}
- Budget level: {request.budget_level}
- Cooking time preference: {request.cooking_time}
- Allergies to exclude: {", ".join(request.allergies) if request.allergies else "None"}
- Health goal: {request.health_goal}
- Variety preference: {request.variety_preference}
- Reuse ingredients across meals when appropriate: {"Yes" if request.reuse_ingredients else "No"}

Rules:
1) Respect all dietary/allergy constraints. Avoid prohibited ingredients.
2) Keep meals simple and beginner-friendly with common ingredients.
3) Provide clear variety according to preference; reuse ingredients if requested.
4) Each meal must include "servings": {request.household_size}.
5) Grocery list must be grouped into categories.

Return ONLY valid JSON matching this exact schema (no markdown, no extra text):

{{
  "weeklyMeals": [
    {{
      "day": "Monday",
      "breakfast": {{"meal": "string", "servings": {request.household_size}}},
      "lunch":     {{"meal": "string", "servings": {request.household_size}}},
      "dinner":    {{"meal": "string", "servings": {request.household_size}}}
    }},
    ... 6 more days through "Sunday"
  ],
  "groceryList": {{
    "Produce": ["item"],
    "Protein": ["item"],
    "Dairy": ["item"],
    "Pantry": ["item"],
    "Spices": ["item"],
    "Other": ["item"]
  }},
  "meta": {{
    "notes": "brief planning notes (optional)"
  }}
}}
"""

    task = Task(
        description=description,
        agent=recipe_agent,
        expected_output="Strict JSON meal plan with 7 days and categorized grocery list.",
    )
    crew = Crew(agents=[recipe_agent], tasks=[task])

    try:
        result = await asyncio.create_task(crew.kickoff_async())
        raw = result.raw or ""
        print("\n=== Raw AI Output ===\n", raw)

        cleaned = _clean_json_text(raw)
        data = json.loads(cleaned)

        # Repair/normalize
        data = _ensure_7_days(data, request.household_size)
        data["groceryList"] = _categorize_grocery_list(data.get("groceryList"))
        data["meta"] = data.get("meta", {})
        return data

    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        # Return raw to help frontend show a useful error
        return {"error": "Invalid JSON from AI", "raw": result.raw if 'result' in locals() else ""}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate-dietplan")
async def generate_dietplan(request: DietPlanRequest):
    # 1) Compute target calories on backend
    bmr = _bmr(request.weight, request.height, request.age, request.gender)
    tdee = _tdee(bmr, request.activity_level)
    if request.target_weight < request.weight:
        # weight loss
        target_cals = round(max(1200, tdee - 500))
        goal = "weight loss"
    elif request.target_weight > request.weight:
        # weight gain
        target_cals = round(tdee + 250)
        goal = "weight gain"
    else:
        target_cals = round(tdee)
        goal = "maintenance"

    # 2) Build strict prompt
    description = f"""
You are a professional nutritionist. Create a personalised 7-day diet plan that fits the user's calorie target and preferences.

User Profile:
- Weight: {request.weight} kg
- Height: {request.height} cm
- Age: {request.age}
- Gender: {request.gender}
- Activity Level: {request.activity_level}
- Goal: {goal} (target daily calories ≈ {target_cals})
- Target Weight: {request.target_weight} kg
- Dietary Preference: {request.dietary_preference}
- Allergies to exclude: {", ".join(request.allergies) if request.allergies else "None"}
- Cuisine Preference: {request.cuisine_preference}
- Cooking Time: {request.cooking_time}
- Budget: {request.budget_level}

Rules:
1) Respect all allergies and dietary preferences. Avoid unsafe ingredients.
2) Keep meals practical and beginner-friendly with common foods.
3) For each day (Monday→Sunday), include breakfast, lunch, dinner, and snack.
4) Assign numeric calories to each meal. Per-day total should be within ±5% of {target_cals}.
5) Keep variety across the week; avoid repeating the exact same meal more than twice.
6) Return ONLY valid JSON. No markdown, no commentary.

JSON schema:
{{
  "dailyCalories": {target_cals},
  "weeklyPlan": [
    {{
      "day": "Monday",
      "breakfast": {{"meal": "string", "calories": number}},
      "lunch":     {{"meal": "string", "calories": number}},
      "dinner":    {{"meal": "string", "calories": number}},
      "snack":     {{"meal": "string", "calories": number}}
    }},
    ... 6 more days through "Sunday"
  ],
  "groceryList": {{
    "Produce": ["item"],
    "Protein": ["item"],
    "Dairy": ["item"],
    "Pantry": ["item"],
    "Spices": ["item"],
    "Other": ["item"]
  }},
  "meta": {{"notes": "optional brief notes"}}
}}
"""

    task = Task(
        description=description,
        agent=recipe_agent,
        expected_output="Strict JSON with 7-day diet plan and categorized grocery list.",
    )
    crew = Crew(agents=[recipe_agent], tasks=[task])

    try:
        result = await asyncio.create_task(crew.kickoff_async())
        raw = result.raw or ""
        print("\n=== Raw AI Output (dietplan) ===\n", raw)

        cleaned = _clean_json_text(raw)
        data = json.loads(cleaned)

        # Repair/normalize to guarantee UI safety
        data = _ensure_7_days_diet(data, target_cals)
        data["groceryList"] = _categorize_grocery_list(data.get("groceryList"))
        data["meta"] = data.get("meta", {})
        # Ensure dailyCalories equals our computed target
        data["dailyCalories"] = int(target_cals)
        return data

    except json.JSONDecodeError as e:
        print("JSON parsing error (dietplan):", e)
        return {"error": "Invalid JSON from AI", "raw": result.raw if 'result' in locals() else ""}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
def ping():
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
