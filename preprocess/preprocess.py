import re
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict


class TextPreprocessor:
    """
    A class to preprocess text files for LLM and AI agent consumption.
    Performs various cleaning and normalization steps to ensure optimal text quality.
    """
    
    def __init__(self, preserve_paragraphs: bool = True, min_line_length: int = 3, verbose: bool = True):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            preserve_paragraphs: Whether to maintain paragraph structure
            min_line_length: Minimum length for a line to be kept
            verbose: Whether to print statistics for each step
        """
        self.preserve_paragraphs = preserve_paragraphs
        self.min_line_length = min_line_length
        self.verbose = verbose
        self.stats = {}
    
    def _print_step_stats(self, step_name: str, original_text: str, processed_text: str, additional_stats: Dict = None):
        """Print statistics for a preprocessing step."""
        if not self.verbose:
            return
            
        char_diff = len(original_text) - len(processed_text)
        line_diff = len(original_text.split('\n')) - len(processed_text.split('\n'))
        
        print(f"\n{step_name}:")
        print(f"  - Characters removed: {char_diff}")
        print(f"  - Lines removed: {line_diff}")
        
        if additional_stats:
            for key, value in additional_stats.items():
                print(f"  - {key}: {value}")

    def load_text_file(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Step 1: Load text file with proper encoding handling.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)
            
        Returns:
            Raw text content from the file
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def normalize_unicode(self, text: str) -> str:
        """
        Step 2: Normalize Unicode characters to standard forms.
        Converts accented characters and special Unicode to normalized forms.
        
        Args:
            text: Input text with potential Unicode issues
            
        Returns:
            Unicode-normalized text
        """
        original_text = text
        # Normalize to NFC (Canonical Decomposition followed by Canonical Composition)
        processed_text = unicodedata.normalize('NFC', text)
        
        # Count Unicode normalizations
        changes = sum(1 for o, p in zip(original_text, processed_text) if o != p)
        self._print_step_stats("Step 2: Unicode Normalization", original_text, processed_text, 
                              {"Characters normalized": changes})
        
        return processed_text
    
    def remove_control_characters(self, text: str) -> str:
        """
        Step 3: Remove control characters that can interfere with LLM processing.
        Keeps only printable characters, newlines, and tabs.
        
        Args:
            text: Input text with potential control characters
            
        Returns:
            Text with control characters removed
        """
        original_text = text
        # Remove control characters except newline (\n), carriage return (\r), and tab (\t)
        processed_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        control_chars_removed = len(original_text) - len(processed_text)
        self._print_step_stats("Step 3: Control Character Removal", original_text, processed_text,
                              {"Control characters removed": control_chars_removed})
        
        return processed_text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Step 4: Normalize whitespace characters.
        Converts various whitespace types to standard spaces and newlines.
        
        Args:
            text: Input text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        original_text = text
        
        # Count multiple spaces before normalization
        multiple_spaces = len(re.findall(r'[ \t]{2,}', text))
        
        # Replace multiple whitespace characters (except newlines) with single spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize line endings to \n
        crlf_count = text.count('\r\n')
        cr_count = text.count('\r') - crlf_count
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remove trailing whitespace from each line
        trailing_spaces = len(re.findall(r'[ \t]+$', text, flags=re.MULTILINE))
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        self._print_step_stats("Step 4: Whitespace Normalization", original_text, text,
                              {"Multiple spaces normalized": multiple_spaces,
                               "CRLF normalized": crlf_count,
                               "CR normalized": cr_count,
                               "Trailing spaces removed": trailing_spaces})
        
        return text
    
    def remove_excessive_newlines(self, text: str) -> str:
        """
        Step 5: Remove excessive newlines while preserving paragraph structure.
        Reduces multiple consecutive newlines to appropriate spacing.
        
        Args:
            text: Input text with potential excessive newlines
            
        Returns:
            Text with normalized newline spacing
        """
        original_text = text
        
        if self.preserve_paragraphs:
            # Count 3+ consecutive newlines
            excessive_newlines = len(re.findall(r'\n{3,}', text))
            # Replace 3+ consecutive newlines with double newlines (paragraph breaks)
            text = re.sub(r'\n{3,}', '\n\n', text)
        else:
            # Count 2+ consecutive newlines
            excessive_newlines = len(re.findall(r'\n{2,}', text))
            # Replace multiple newlines with single newlines
            text = re.sub(r'\n{2,}', '\n', text)
        
        self._print_step_stats("Step 5: Excessive Newline Removal", original_text, text,
                              {"Excessive newline groups normalized": excessive_newlines})
        
        return text
    
    def remove_mid_sentence_newlines(self, text: str) -> str:
        """
        Step 5.5: Remove ALL newlines and add them only after sentence-ending punctuation.
        Aggressively removes all newlines, then adds newlines after ., !, ?, :, ;
        
        Args:
            text: Input text with potential mid-sentence newlines
            
        Returns:
            Text with newlines only after sentence endings
        """
        original_text = text
        
        # Count original newlines for stats
        original_newlines = text.count('\n')
        
        # Step 1: Remove ALL newlines and replace with spaces
        text_no_newlines = re.sub(r'\n+', ' ', text)
        
        # Step 2: Clean up multiple spaces
        text_no_newlines = re.sub(r'\s+', ' ', text_no_newlines).strip()
        
        # Step 3: Add newlines after sentence-ending punctuation
        # Look for sentence endings (., !, ?, :, ;) followed by space and then any character
        processed_text = re.sub(
            r'([.!?:;])\s+', 
            r'\1\n', 
            text_no_newlines
        )
        
        # Step 4: Handle special cases - don't break on abbreviations or decimals
        # Restore spaces for common abbreviations (Mr., Dr., etc.)
        processed_text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|vs|etc|i\.e|e\.g)\n', r'\1. ', processed_text)
        
        # Restore spaces for decimal numbers
        processed_text = re.sub(r'(\d)\n(\d)', r'\1.\2', processed_text)
        
        # Step 5: Clean up any trailing spaces before newlines
        processed_text = re.sub(r' +\n', '\n', processed_text)
        
        # Step 6: Ensure paragraphs are separated by double newlines
        # Look for places where sentences should start new paragraphs
        # (This is optional - you can remove this if you want single newlines only)
        processed_text = re.sub(r'\n([A-Z][a-z]+ \d+,)', r'\n\n\1', processed_text)  # "On day X,"
        
        # Count final newlines
        final_newlines = processed_text.count('\n')
        newlines_removed = original_newlines - final_newlines
        
        self._print_step_stats("Step 5.5: Mid-sentence Newline Removal", original_text, processed_text,
                              {"Total newlines removed": newlines_removed,
                               "Sentences properly ended": final_newlines})
        
        return processed_text

    def remove_empty_lines(self, text: str) -> str:
        """
        Step 6: Remove empty lines and lines with only whitespace.
        
        Args:
            text: Input text with potential empty lines
            
        Returns:
            Text with empty lines removed
        """
        original_text = text
        lines = text.split('\n')
        original_line_count = len(lines)
        
        # Keep only non-empty lines that meet minimum length requirement
        filtered_lines = [
            line for line in lines 
            if line.strip() and len(line.strip()) >= self.min_line_length
        ]
        
        empty_lines_removed = original_line_count - len(filtered_lines)
        processed_text = '\n'.join(filtered_lines)
        
        self._print_step_stats("Step 6: Empty Line Removal", original_text, processed_text,
                              {"Empty/short lines removed": empty_lines_removed})
        
        return processed_text

    def fix_common_encoding_issues(self, text: str) -> str:
        """
        Step 7: Fix common encoding artifacts and character issues.
        
        Args:
            text: Input text with potential encoding issues
            
        Returns:
            Text with encoding issues resolved
        """
        original_text = text
        
        # Common encoding fixes
        replacements = {
            'â€™': "'",  # Smart apostrophe
            'â€œ': '"',  # Smart quote open
            'â€': '"',   # Smart quote close
            'â€"': '—',  # Em dash
            'â€"': '–',  # En dash
            'Â': '',     # Non-breaking space artifacts
            'â€¦': '...', # Ellipsis
        }
        
        total_fixes = 0
        for old, new in replacements.items():
            count = text.count(old)
            total_fixes += count
            text = text.replace(old, new)
        
        self._print_step_stats("Step 7: Encoding Issue Fixes", original_text, text,
                              {"Encoding artifacts fixed": total_fixes})
        
        return text

    def clean_special_characters(self, text: str) -> str:
        """
        Step 8: Clean or normalize special characters for better LLM processing.
        
        Args:
            text: Input text with special characters
            
        Returns:
            Text with special characters cleaned
        """
        original_text = text
        
        # Count smart quotes
        smart_double_quotes = len(re.findall(r'["""]', text))
        smart_single_quotes = len(re.findall(r"[''']", text))
        
        # Replace smart quotes with regular quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Count special dashes
        special_dashes = len(re.findall(r'[–—]', text))
        # Replace various dash types with standard hyphens/dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Count ellipsis characters
        ellipsis_chars = text.count('…')
        # Replace ellipsis character with three dots
        text = text.replace('…', '...')
        
        self._print_step_stats("Step 8: Special Character Cleaning", original_text, text,
                              {"Smart double quotes normalized": smart_double_quotes,
                               "Smart single quotes normalized": smart_single_quotes,
                               "Special dashes normalized": special_dashes,
                               "Ellipsis characters normalized": ellipsis_chars})
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Main preprocessing pipeline that applies all cleaning steps in sequence.
        
        Args:
            text: Raw input text
            
        Returns:
            Fully preprocessed text ready for LLM processing
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STARTING TEXT PREPROCESSING")
            print("="*60)
            print(f"Original text length: {len(text)} characters")
            print(f"Original line count: {len(text.split('\n'))}")
        
        # Apply all preprocessing steps in order
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.fix_common_encoding_issues(text)
        text = self.clean_special_characters(text)
        text = self.normalize_whitespace(text)
        text = self.remove_excessive_newlines(text)
        text = self.remove_mid_sentence_newlines(text)  # New step added here
        text = self.remove_empty_lines(text)
        
        # Final cleanup: strip leading/trailing whitespace
        original_length = len(text)
        text = text.strip()
        stripped_chars = original_length - len(text)
        
        if self.verbose:
            print(f"\nFinal cleanup - Leading/trailing whitespace removed: {stripped_chars}")
            print("\n" + "="*60)
            print("PREPROCESSING COMPLETE")
            print("="*60)
            print(f"Final text length: {len(text)} characters")
            print(f"Final line count: {len(text.split('\n'))}")
            print(f"Total characters removed: {len(self.stats.get('original_text', '')) - len(text) if 'original_text' in self.stats else 'N/A'}")
        
        return text
    
    def preprocess_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Complete preprocessing workflow: load file, preprocess, and optionally save.
        
        Args:
            file_path: Path to input text file
            output_path: Optional path to save preprocessed text
            
        Returns:
            Preprocessed text content
        """
        # Load the text file
        raw_text = self.load_text_file(file_path)
        
        # Preprocess the text
        processed_text = self.preprocess_text(raw_text)
        
        # Save to output file if specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(processed_text)
        
        return processed_text


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor with verbose output
    preprocessor = TextPreprocessor(preserve_paragraphs=True, min_line_length=3, verbose=True)
    
    # Process the case1.txt file
    input_file = "/home/gusmmm/Desktop/demos/demo_basic_structured_output/texts/case1.txt"
    
    try:
        print("Processing case1.txt...")
        
        # Store original text for comparison
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        preprocessor.stats['original_text'] = original_text
        
        # Preprocess the file
        processed_text = preprocessor.preprocess_file(input_file)
        
        # Print the preprocessed text
        print("\n\nPREPROCESSED TEXT:")
        print("-" * 60)
        print(processed_text)
        print("-" * 60)
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
        print("-" * 60)
        print(processed_text)
        print("-" * 60)
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
