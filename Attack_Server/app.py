"""
Attack Server (Main Server) - WITH ITERATION STRING TRACKING (~ 465 lines)
Key Addition: Tracks the updated identifier string after each iteration,
showing progression and the top-ranked car at each step.
"""
from fastapi import FastAPI, HTTPException
# from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import httpx

app = FastAPI(title="Attack Server (Main)")

# Serve static files
app.mount(
    "/static",
    StaticFiles(directory="./static", html=True),
    name="static",
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "../OBD.db"
CAR_MODEL_SERVER_URL = "http://localhost:8001"

# --- Models ---
class IterativeAttackRequest(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None

class FieldTestAttempt(BaseModel):
    field: str
    candidate_car: str
    candidate_rank: int
    test_value: str
    matched: bool
    message: str

class TopRankedCar(BaseModel):
    rank: int
    manufacturer: str
    model: str
    year: Optional[int]
    exact_matches: int
    similarity: float
    identifier: str

class IterationDetail(BaseModel):
    iteration_number: int
    field: str
    attempts: List[FieldTestAttempt]
    final_status: str
    matched_car: Optional[str] = None
    matched_value: Optional[str] = None
    updated_string: str
    top_ranked_car: TopRankedCar
    list_resorted: bool
    resorted_list: Optional[List[Dict[str, Any]]] = None

class IterativeAttackResponse(BaseModel):
    target_car_info: Dict[str, Any]
    initial_identifier: str
    initial_sorted_list: List[Dict[str, Any]]
    iterations: List[IterationDetail]
    final_string: str
    final_matched_fields: Dict[str, bool]
    attack_summary: Dict[str, Any]

# --- Domain & Loaders ---
class Car:
    def __init__(self, manufacturer: str, model: str, year: int = None, 
                 country_region: str = "", type_: str = ""):
        self.manufacturer = manufacturer
        self.model = model
        self.year = year
        self.country_region = country_region
        self.type_ = type_
        self.commands: Dict[str, List[Dict[str, Any]]] = {}

    def add_command(self, field: str, raw_json: Any):
        parsed = (
            json.loads(raw_json)
            if isinstance(raw_json, str)
            else raw_json
        )

        cmd_list: List[Dict[str, Any]]
        if isinstance(parsed, dict):
            cmd_list = [parsed]
        elif isinstance(parsed, list):
            cmd_list = [p for p in parsed if isinstance(p, dict)]
        else:
            return

        self.commands.setdefault(field, []).extend(cmd_list)

def load_all_known_cars() -> List[Car]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT manufacturer, model, year, country_region, type,
               Pack_Voltage, Pack_SOC, Pack_SOH
        FROM vehicle_pack_commands
    """)
    rows = cur.fetchall()
    conn.close()
    
    cars_map: Dict[Tuple[str, str, int, str, str], Car] = {}

    for (manufacturer, model, year, country_region, type_,
         raw_vol, raw_soc, raw_soh) in rows:
        key = (manufacturer, model, year, country_region, type_)

        if key not in cars_map:
            cars_map[key] = Car(manufacturer, model, year, country_region, type_)

        car = cars_map[key]

        if raw_vol:
            car.add_command("Pack_Voltage", raw_vol)
        if raw_soc:
            car.add_command("Pack_SOC", raw_soc)
        if raw_soh:
            car.add_command("Pack_SOH", raw_soh)

    return list(cars_map.values())

def canonical_identifier(manufacturer: Optional[str] = None, 
                        model: Optional[str] = None,
                        year: Optional[int] = None, 
                        country_region: Optional[str] = None, 
                        type_: Optional[str] = None) -> str:
    if not manufacturer and not model:
        identifier = "unknown"
    else:
        m = (manufacturer or "").strip().lower().replace(" ", "")
        mod = (model or "").strip().lower().replace(" ", "")
        
        if m and mod:
            identifier = f"{m}_{mod}"
        elif m:
            identifier = m
        elif mod:
            identifier = mod
        else:
            identifier = "unknown"

    if year is not None:
        identifier += f"_{year}"
    if country_region:
        identifier += f"_{country_region.strip().lower().replace(' ', '')}"
    if type_:
        identifier += f"_{type_.strip().lower().replace(' ', '')}"
    
    return identifier

def stringify_commands(cmd_list: List[Dict[str, Any]]) -> str:
    if not cmd_list:
        return ""
    return json.dumps(cmd_list, separators=(',', ':'))

def build_car_identifier_with_commands(car: Car, fields_to_include: List[str]) -> str:
    base_id = canonical_identifier(
        car.manufacturer,
        car.model,
        car.year,
        car.country_region,
        car.type_
    )
    
    for field in fields_to_include:
        if field in car.commands and car.commands[field]:
            cmd_str = stringify_commands(car.commands[field])
            base_id += f"_{field}:{cmd_str}"
    
    return base_id

def calculate_similarity_scores(target_identifier: str, 
                                car_identifiers: List[str]) -> np.ndarray:
    all_ids = car_identifiers + [target_identifier]
    
    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            max_df=1.0
        )
        vectorizer.fit(all_ids)
        target_vec = vectorizer.transform([target_identifier])
        car_vecs = vectorizer.transform(car_identifiers)
        similarities = cosine_similarity(target_vec, car_vecs).flatten()
        return similarities
    except:
        return np.ones(len(car_identifiers))

def count_exact_field_matches(car: Car, matched_fields_values: Dict[str, str]) -> int:
    """
    Count how many of the matched fields this car has exact values for.
    """
    exact_matches = 0
    
    for field, matched_value_str in matched_fields_values.items():
        if field in car.commands and car.commands[field]:
            car_value_str = stringify_commands(car.commands[field])
            if car_value_str == matched_value_str:
                exact_matches += 1
    
    return exact_matches

# --- Car Model Server Communication ---
async def get_target_car_info():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CAR_MODEL_SERVER_URL}/car-info")
        response.raise_for_status()
        return response.json()

async def test_field_with_car_model(field: str, value: List[Dict[str, Any]]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CAR_MODEL_SERVER_URL}/test-field",
            json={"field": field, "value": value}
        )
        response.raise_for_status()
        return response.json()

# --- Main Attack Endpoint ---
@app.post("/iterative-attack", response_model=IterativeAttackResponse)
async def iterative_guessing_attack(request: IterativeAttackRequest):
    """
    Iterative attack with string tracking after each iteration.
    Shows progression: Iteration 0 (initial), Iteration 1, 2, 3... with updated strings
    """
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]
    
    # Step 1: Fetch target car info
    try:
        target_car_info = await get_target_car_info()
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch target car info: {str(e)}")
    
    # Step 2: Create initial identifier
    initial_identifier = canonical_identifier(
        request.manufacturer,
        request.model,
        request.year,
        request.country_region,
        request.type_
    )
    
    # Step 3: Load all known cars
    known_cars = load_all_known_cars()
    if not known_cars:
        raise HTTPException(500, detail="No cars in database")
    
    # Create base car identifiers (never changes)
    car_base_identifiers = []
    for car in known_cars:
        car_id = canonical_identifier(
            car.manufacturer,
            car.model,
            car.year,
            car.country_region,
            car.type_
        )
        car_base_identifiers.append(car_id)
    
    # Calculate base similarities (never changes)
    base_similarities = calculate_similarity_scores(initial_identifier, car_base_identifiers)
    
    # Create initial match list
    initial_matches = []
    for idx, car in enumerate(known_cars):
        initial_matches.append({
            "car": car,
            "car_base_identifier": car_base_identifiers[idx],
            "base_similarity": float(base_similarities[idx]),
            "exact_match_count": 0,
            "rank": 0
        })
    
    # Sort by initial similarity only
    initial_matches.sort(key=lambda x: x["base_similarity"], reverse=True)
    
    # Assign initial ranks
    for i, match in enumerate(initial_matches):
        match["rank"] = i + 1
    
    # Save initial top 20
    initial_list_info = [
        {
            "rank": match["rank"],
            "manufacturer": match["car"].manufacturer,
            "model": match["car"].model,
            "year": match["car"].year,
            "similarity": match["base_similarity"],
            "identifier": match["car_base_identifier"]
        }
        for match in initial_matches[:20]
    ]
    
    # Step 4: Iterative field testing with string tracking
    current_identifier = initial_identifier
    matched_fields_values = {}
    iterations = []
    final_matched_fields = {}
    total_attempts = 0
    
    for field_idx, field in enumerate(ALL_PARAMS):
        iteration_number = field_idx + 1
        field_attempts = []
        field_matched = False
        matched_car_name = None
        matched_value = None
        list_was_resorted = field_idx > 0 and len(matched_fields_values) > 0
        
        # Capture current sorted list (top 20) before testing this field
        current_sorted_list = [
            {
                "rank": match["rank"],
                "manufacturer": match["car"].manufacturer,
                "model": match["car"].model,
                "year": match["car"].year,
                "exact_matches": match["exact_match_count"],
                "similarity": match["base_similarity"],
                "identifier": build_car_identifier_with_commands(match["car"], list(matched_fields_values.keys()))
            }
            for match in initial_matches[:20]
        ] if list_was_resorted else None
        
        # Test each candidate in current order
        for match_info in initial_matches:
            car = match_info["car"]
            
            if field not in car.commands or not car.commands[field]:
                continue
            
            test_value = car.commands[field]
            test_value_str = stringify_commands(test_value)
            
            # Test with Car Model Server
            total_attempts += 1
            try:
                test_response = await test_field_with_car_model(field, test_value)
                matched = test_response["matched"]
                
                attempt = FieldTestAttempt(
                    field=field,
                    candidate_car=f"{car.manufacturer} {car.model} {car.year or ''}".strip(),
                    candidate_rank=match_info["rank"],
                    test_value=test_value_str,
                    matched=matched,
                    message=test_response["message"]
                )
                field_attempts.append(attempt)
                
                # If matched, update string and re-sort
                if matched:
                    field_matched = True
                    matched_car_name = f"{car.manufacturer} {car.model}"
                    matched_value = test_value_str
                    
                    # Update identifier string
                    current_identifier += f"_{field}:{test_value_str}"
                    matched_fields_values[field] = test_value_str
                    
                    # Update exact match counts for ALL cars
                    for match in initial_matches:
                        match["exact_match_count"] = count_exact_field_matches(
                            match["car"],
                            matched_fields_values
                        )
                    
                    # Sort by exact match count (PRIMARY), then base similarity (TIEBREAKER)
                    initial_matches.sort(
                        key=lambda x: (x["exact_match_count"], x["base_similarity"]),
                        reverse=True
                    )
                    
                    # Update ranks
                    for i, match in enumerate(initial_matches):
                        match["rank"] = i + 1
                    
                    break
                
            except Exception as e:
                attempt = FieldTestAttempt(
                    field=field,
                    candidate_car=f"{car.manufacturer} {car.model} {car.year or ''}".strip(),
                    candidate_rank=match_info["rank"],
                    test_value=test_value_str,
                    matched=False,
                    message=f"Error: {str(e)}"
                )
                field_attempts.append(attempt)
        
        # Get top-ranked car after this iteration
        top_car = initial_matches[0]
        top_ranked_car = TopRankedCar(
            rank=1,
            manufacturer=top_car["car"].manufacturer,
            model=top_car["car"].model,
            year=top_car["car"].year,
            exact_matches=top_car["exact_match_count"],
            similarity=top_car["base_similarity"],
            identifier=build_car_identifier_with_commands(top_car["car"], list(matched_fields_values.keys()))
        )
        
        # Log iteration results with updated string
        iteration_detail = IterationDetail(
            iteration_number=iteration_number,
            field=field,
            attempts=field_attempts,
            final_status="matched" if field_matched else "not_found",
            matched_car=matched_car_name,
            matched_value=matched_value,
            updated_string=current_identifier,
            top_ranked_car=top_ranked_car,
            list_resorted=list_was_resorted,
            resorted_list=current_sorted_list
        )
        iterations.append(iteration_detail)
        final_matched_fields[field] = field_matched
    
    # Calculate statistics
    matched_count = sum(1 for matched in final_matched_fields.values() if matched)
    attack_summary = {
        "total_fields": len(ALL_PARAMS),
        "fields_matched": matched_count,
        "fields_not_found": len(ALL_PARAMS) - matched_count,
        "total_attempts": total_attempts,
        "success_rate": f"{matched_count}/{len(ALL_PARAMS)}",
        "average_attempts_per_field": total_attempts / len(ALL_PARAMS) if ALL_PARAMS else 0
    }
    
    return IterativeAttackResponse(
        target_car_info=target_car_info,
        initial_identifier=initial_identifier,
        initial_sorted_list=initial_list_info,
        iterations=iterations,
        final_string=current_identifier,
        final_matched_fields=final_matched_fields,
        attack_summary=attack_summary
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)