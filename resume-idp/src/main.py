from src.data_loader.resume_loader import load_resumes
from src.pipeline.extractor import extract_from_resumes
from src.config.settings import DATA_PATH

def main():
    print("📥 Loading resume dataset...")
    df = load_resumes(DATA_PATH)

    print("🧠 Extracting structured information using Ollama...")
    extract_from_resumes(df)

    print("✅ Extraction completed.")

if __name__ == "__main__":
    main()


