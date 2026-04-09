from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.dependencies import get_current_user
from core.database import get_db
from models.user import User
from models.workspace import Workspace
from models.version import Version
from pydantic import BaseModel
import time
from api.utils.transliteration import process_transliteration

router = APIRouter()

class TranslateSchema(BaseModel):
    workspaceId: str
    input: str
    targetDialect: str = "hindi"

class TransliterateSchema(BaseModel):
    text: str
    targetDialect: str
    workspaceId: str

class EditLyricsSchema(BaseModel):
    workspaceId: str
    targetWord: str
    context: dict
    customPrompt: str

class GenerateRhymeSchema(BaseModel):
    workspaceId: str
    lines: list[str]
    genre: str = "party"
    rhymeScheme: str = "AABB"

from fastapi.responses import StreamingResponse
import json

@router.post("/generate-rhyme")
def generate_rhyme(data: GenerateRhymeSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == data.workspaceId, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    import requests
    import re
    from core.config import settings

    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.HF_API_KEY
    }
    
    # Filter out empty lines to avoid sending bad data to model
    valid_lines = [line.strip() for line in data.lines if line.strip()]
    
    if not valid_lines:
        return {"rhymedLines": []}

    def get_last_word(text):
        words = re.sub(r'[।,.!?"\']', '', text).split()
        return words[-1] if words else 'none'

    generated_lines = []
    prev_end = 'none'
    rhyme_scheme = "AABB"
    backend_genre = data.genre.lower()

    for i, rough in enumerate(valid_lines):
        rg = rhyme_scheme[i % len(rhyme_scheme)]
        hf_payload = {
            "genre": backend_genre,
            "rhyme_scheme": rg,
            "rough_lines": [rough],
            "prev_end": prev_end
        }

        try:
            resp = requests.post(settings.HF_POETRY_URL, json=hf_payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                result_verses = resp.json().get("verse", [])
                line_result = result_verses[0] if result_verses else rough + " (Empty Response)"
            else:
                line_result = rough + f" (Error {resp.status_code})"
        except Exception as e:
            line_result = rough + f" (Error {str(e)})"
            
        generated_lines.append(line_result)
        prev_end = get_last_word(line_result)
        
    # Create an implicit version to track Rhymed output
    new_output = "\n".join(generated_lines)
    new_version = Version(
        workspace_id=workspace.id,
        input_state=workspace.current_input,
        output_state=new_output,
        label=f"Rhyme Generated ({data.genre}, {data.rhymeScheme})"
    )
    db.add(new_version)
    
    workspace.current_output = new_output
    db.commit()

    return {"rhymedLines": generated_lines}


@router.post("/translate")
def translate(data: TranslateSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == data.workspaceId, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    code_map = {
        "hindi": "hin_Deva",
        "marathi": "mar_Deva",
        "english": "eng_Latn"
    }
    tgt_lang = code_map.get(data.targetDialect.lower(), "hin_Deva")
    src_lang = "eng_Latn"

    def generate_translations():
        import requests
        from core.config import settings
        
        lines = [line.strip() for line in data.input.split("\n")]
        full_translated_lines = []

        headers = {
            "Content-Type": "application/json",
            "x-api-key": settings.HF_API_KEY
        }

        for line in lines:
            if not line:
                result = ""
            else:
                hf_payload = {
                    "query": line,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang
                }
                try:
                    resp = requests.post(settings.HF_TRANSLATE_URL, json=hf_payload, headers=headers, timeout=20)
                    result = resp.json().get("translation", line) if resp.status_code == 200 else (line + " (Error)")
                except Exception:
                    result = line + " (Error)"
            
            full_translated_lines.append(result)
            yield json.dumps({"line": result}) + "\n"

        # Finalize DB state after streaming is done
        mock_output = "\n".join(full_translated_lines)
        new_version = Version(
            workspace_id=workspace.id,
            input_state=data.input,
            output_state=mock_output,
            label=f"Translation to {data.targetDialect}"
        )
        db.add(new_version)
        workspace.current_input = data.input
        workspace.current_output = mock_output
        db.commit()

    return StreamingResponse(generate_translations(), media_type="application/x-ndjson")


@router.post("/transliterate")
def transliterate(data: TransliterateSchema, current_user: User = Depends(get_current_user)):
    lines = data.text.split("\n")
    transliterated_lines = []
    
    for line in lines:
        if not line.strip():
            transliterated_lines.append("")
            continue
            
        result = process_transliteration(line, data.targetDialect)
        # Prioritize 'transliteration' (Romanized) over 'pronunciation_english' (IPA)
        # as it is more stable and what users typically expect for phonetics.
        out_text = result["transliteration"] if result["transliteration"] else result["pronunciation_english"]
        transliterated_lines.append(out_text)
        
    return {"transliteratedLines": transliterated_lines}

@router.post("/edit-lyrics")
def edit_lyrics(data: EditLyricsSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == data.workspaceId, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    line_index = data.context.get("lineIndex", 0)
    word_index = data.context.get("wordIndex", 0)
    
    lines = (workspace.current_output or "").split('\n')
    
    if 0 <= line_index < len(lines):
        words = lines[line_index].split(' ')
        if 0 <= word_index < len(words):
            words[word_index] = data.targetWord
            lines[line_index] = " ".join(words)
            
    new_output = "\n".join(lines)
    
    # Implicitly create a version for the edit (Beam Search Mock)
    new_version = Version(
        workspace_id=workspace.id,
        input_state=workspace.current_input,
        output_state=new_output,
        label=f"Beam search update: '{data.targetWord}'"
    )
    db.add(new_version)
    
    workspace.current_output = new_output
    db.commit()
    
    return {"newOutput": new_output}
