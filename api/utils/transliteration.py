from unidecode import unidecode
from aksharamukha import transliterate
import epitran
import logging

logger = logging.getLogger(__name__)

EPITRAN_LANGUAGE_MAP = {
    "hi": "hin-Deva",
    "mr": "mar-Deva",
    "bn": "ben-Beng",
    "ta": "tam-Taml",
    "te": "tel-Telu",
    "ml": "mal-Mlym",
    "kn": "kan-Knda",
    "gu": "guj-Gujr",
    "pa": "pan-Guru"
}

LANGUAGE_TO_ISO = {
    "hindi": "hi",
    "marathi": "mr",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "kannada": "kn",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "assamese": "as"
}

def transliterate_text(text: str, language_code: str = None) -> str:
    try:
        # We'll try to use autodetect for the source script to be more robust
        # Aksharamukha often works better if the source is explicitly 'autodetect' 
        # when passing mixed content or different variants.
        
        return transliterate.process(
            'autodetect',
            'ISO',
            text
        )

    except Exception as e:
        logger.error(f"Aksharamukha transliteration failed: {e}")
        try:
            return unidecode(text)
        except:
            return text

def normalize_to_simple_english(text: str, language_code: str) -> str:
    # Common IPA / scholarly replacements
    replacements = {
        "ā": "aa",
        "ī": "ee",
        "ū": "oo",
        "ē": "e",
        "ō": "o",
        "ə": "",
        "ɦ": "h",
        "ʈ": "t",
        "ɖ": "d",
        "ɳ": "n",
        "ŋ": "ng",
        "ɲ": "ny",
        "ʃ": "sh",
        "ʂ": "sh",
        "ː": "",
        "ṃ": "m",
        "ṁ": "m",
        "ṅ": "ng",
        "ñ": "ny",
        "ṇ": "n",
        "t̪": "t",
        "d̪": "d",
        "b̤": "bh",
        "l̪": "l",
        "ɔ": "o",
        "e̯": "",
        "ɯ": "u",
        "ɭ": "l",
        "ɽ": "r",
    }

    result = text

    # Global replacements first
    for old, new in replacements.items():
        result = result.replace(old, new)

    # ---------- Hindi / Marathi / Sanskrit / Nepali ----------
    if language_code in ["hi", "mr", "sa", "ne"]:
        words = result.split()
        cleaned_words = []

        for word in words:
            # remove schwa ending
            if word.endswith("a") and len(word) > 3:
                word = word[:-1]

            # better common sounds
            word = (
                word.replace("bh", "bh")
                    .replace("ph", "ph")
                    .replace("kh", "kh")
                    .replace("gh", "gh")
            )

            cleaned_words.append(word)

        result = " ".join(cleaned_words)

    # ---------- Bengali / Assamese ----------
    elif language_code in ["bn", "as"]:
        result = (
            result
            .replace("aa", "a")
            .replace("ẏa", "y")
            .replace("ẏ", "y")
            .replace("basi", "bashi")
            .replace("s", "sh")
            .replace("tomaa", "toma")
        )

    # ---------- Tamil ----------
    elif language_code == "ta":
        result = (
            result
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
            .replace("zh", "zh")   # special Tamil sound
            .replace("th", "th")
        )

    # ---------- Telugu ----------
    elif language_code == "te":
        result = (
            result
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
            .replace("lu", "lu")
            .replace("du", "du")
        )

    # ---------- Kannada ----------
    elif language_code == "kn":
        result = (
            result
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
        )

    # ---------- Malayalam ----------
    elif language_code == "ml":
        result = (
            result
            .replace("nj", "nj")
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
        )

    # ---------- Gujarati ----------
    elif language_code == "gu":
        result = (
            result
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
        )

    # ---------- Punjabi ----------
    elif language_code == "pa":
        result = (
            result
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
            .replace("kh", "kh")
            .replace("gh", "gh")
        )

    # ---------- Odia ----------
    elif language_code == "or":
        result = (
            result
            .replace("aa", "aa")
            .replace("ii", "ee")
            .replace("uu", "oo")
        )

    return " ".join(result.split()).lower().strip()

EPITRAN_CACHE = {}

def get_pronunciation(text: str, language_code: str) -> str:
    try:
        code = EPITRAN_LANGUAGE_MAP.get(language_code)

        if not code:
            return text

        if code not in EPITRAN_CACHE:
            logger.info(f"Initializing Epitran for {code}...")
            EPITRAN_CACHE[code] = epitran.Epitran(code)
            
        epi = EPITRAN_CACHE[code]
        return epi.transliterate(text)

    except Exception as e:
        logger.error(f"Epitran transliteration failed for {language_code}: {e}")
        return text

def process_transliteration(text: str, language: str) -> dict:
    # Map long name to code
    language_code = LANGUAGE_TO_ISO.get(language.lower(), "hi")
    
    transliterated = transliterate_text(text, language_code)
    pronunciation = get_pronunciation(text, language_code)

    transliterated = normalize_to_simple_english(
        transliterated,
        language_code
    )

    pronunciation = normalize_to_simple_english(
        pronunciation,
        language_code
    )

    return {
        "original_text": text,
        "language_detected": language_code,
        "transliteration": transliterated,
        "pronunciation_english": pronunciation
    }
