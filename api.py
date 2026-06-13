import os

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Stack Overflow Developer API")

DATA_FILE = "cleaned_survey_data.csv"


class Developer(BaseModel):
    ConvertedCompYearly: float
    WorkExp: float
    YearsCodeHobby: float
    EdLevel_Numeric: float
    ToolCountWork: float
    ToolCountPersonal: float
    AISelect: str
    ICorPM: str


@app.get("/api/data")
def get_developers(
    skip: int = Query(0, description="Pagination: rows to skip"),
    limit: int = Query(50, description="Pagination: max rows to return"),
    min_salary: float = Query(0.0, description="Filter: Minimum yearly compensation"),
    min_experience: float = Query(
        0.0, description="Filter: Minimum years of professional experience"
    ),
):
    if not os.path.exists(DATA_FILE):
        return {"data": [], "message": "Cleaned data file not found."}

    df = pd.read_csv(DATA_FILE)

    filtered_df = df[
        (df["ConvertedCompYearly"] >= min_salary) & (df["WorkExp"] >= min_experience)
    ]

    paginated_df = filtered_df.iloc[skip : skip + limit]

    result = paginated_df.to_dict(orient="records")

    return {"total_matches": len(filtered_df), "returned": len(result), "data": result}


@app.post("/api/data", status_code=201)
def add_developer(dev: Developer):
    if not os.path.exists(DATA_FILE):
        raise HTTPException(
            status_code=404, detail="Cleaned data file not found. Run Notebook first."
        )

    df = pd.read_csv(DATA_FILE)

    new_data = dev.model_dump()
    new_df = pd.DataFrame([new_data])
    df = pd.concat([df, new_df], ignore_index=True)

    try:
        df.to_csv(DATA_FILE, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

    return {"message": "New developer instance created successfully!", "data": new_data}
