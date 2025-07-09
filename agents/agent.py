from google import genai
from google.genai import types
from pydantic import BaseModel
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the preprocess directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "preprocess"))

from preprocess import TextPreprocessor

class Diagnosis(BaseModel):
    """
    A class to represent a diagnostic term in a NL medical text.
    """
    term: str # The term itself, e.g., "diabetes"
    context: str # The context of the term in the text, e.g., "patient has diabetes"
    temporal: str # The temporal aspect, e.g., "present", "past", "chronic"

class DiagnosisList(BaseModel):
    """
    A class to represent a list of diagnostic terms.
    """
    diagnostics: list[Diagnosis] # A list of Diagnostic objects



def main():
    """Main function to process case1.txt and print results."""
    
    print("🔧 Initializing text preprocessor...")
    # Initialize the text preprocessor with verbose=False for cleaner output
    preprocessor = TextPreprocessor(preserve_paragraphs=True, min_line_length=3, verbose=False)
    
    # Define file paths
    case_file = Path(__file__).parent.parent / "texts" / "case1.txt"
    output_dir = Path(__file__).parent.parent / "output"
    output_file = output_dir / "diagnosis_extraction.json"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    try:
        print(f"📄 Loading and processing file: {case_file.name}")
        print("=" * 60)
        
        # Load and preprocess the file
        processed_text = preprocessor.preprocess_file(str(case_file))
        
        print("✅ Text preprocessing completed successfully!")
        print(f"📊 Processing statistics:")
        print(f"   • Total characters: {len(processed_text)}")
        print(f"   • Total words: {len(processed_text.split())}")
        print(f"   • Total lines: {len(processed_text.split(chr(10)))}")
        print(f"   • Total paragraphs: {len([p for p in processed_text.split(chr(10)*2) if p.strip()])}")
        
        print("\n🤖 Initializing Gemini AI client...")
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        client = genai.Client(api_key=api_key)
        
        print("🔍 Extracting medical diagnoses using AI...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", 
            contents="Extract all medical diagnoses from the following text. Return a JSON list of diagnoses with their context and temporal aspects.\n\n" + processed_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiagnosisList,
                temperature=0.1
            ),
        )
        
        print("✅ AI processing completed!")
        
        # Parse the response
        response_data = json.loads(response.text)
        
        # Save to JSON file
        print(f"💾 Saving results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        # Print readable output to terminal
        print("\n" + "="*60)
        print("🏥 EXTRACTED MEDICAL DIAGNOSES")
        print("="*60)
        
        if 'diagnostics' in response_data and response_data['diagnostics']:
            for i, diagnosis in enumerate(response_data['diagnostics'], 1):
                print(f"\n📋 Diagnosis #{i}:")
                print(f"   🔸 Term: {diagnosis.get('term', 'N/A')}")
                print(f"   🔸 Context: {diagnosis.get('context', 'N/A')}")
                print(f"   🔸 Temporal: {diagnosis.get('temporal', 'N/A')}")
                print("-" * 40)
            
            print(f"\n📈 Summary: Found {len(response_data['diagnostics'])} medical diagnoses")
        else:
            print("⚠️  No diagnoses found in the text.")
        
        print(f"\n✅ Results saved to: {output_file}")
        print("🎉 Process completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {case_file}")
        print("Please ensure the file exists in the texts directory.")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing AI response as JSON: {e}")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
