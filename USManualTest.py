"""
USManualTest.py
CLI entry point for manual test generation.
"""

import os
import sys
import configparser
import pandas as pd
from colorama import init, Fore, Style
from us_to_mtc_file.GenerateManualTest import ManualTestProcessor

# Initialize colorama for Windows
init(autoreset=True)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV or Excel file with encoding fallback.

    Args:
        file_path: Path to the input file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If file cannot be loaded.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".xlsx", ".xls"]:
        # Excel file
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file {file_path}: {str(e)}")
    elif file_ext == ".csv":
        # CSV file with encoding fallback
        for encoding in ["utf-8", "latin-1"]:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ValueError(f"Failed to load CSV file {file_path}: {str(e)}")
        raise ValueError(f"Failed to load CSV file {file_path} with any encoding")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv, .xlsx, or .xls")


def main():
    """Main CLI entry point."""
    print(Fore.CYAN + Style.BRIGHT + "=" * 80)
    print(Fore.CYAN + Style.BRIGHT + "User Story to Manual Test Case Generator (CLI)")
    print(Fore.CYAN + Style.BRIGHT + "=" * 80)
    print()

    try:
        # Load configuration
        config_io = configparser.ConfigParser()
        config_io.read("Config/ConfigIO.properties")

        input_file_path = config_io.get("Input", "input_file_path")
        additional_context_path = config_io.get("Input", "additional_context_path")
        output_file_path = config_io.get("Output", "output_file_path")
        manual_test_type = config_io.get("Output", "manual_test_type")

        print(Fore.YELLOW + "Configuration loaded:")
        print(f"  Input file: {input_file_path}")
        print(f"  Additional context: {additional_context_path}")
        print(f"  Output directory: {output_file_path}")
        print(f"  Test type: {manual_test_type}")
        print()

        # Load user stories
        print(Fore.CYAN + "Loading user stories...")
        input_us_df = load_dataframe(input_file_path)
        print(Fore.GREEN + f"✓ Loaded {len(input_us_df)} user stories")
        print()

        # Initialize processor
        processor = ManualTestProcessor()

        # Check if additional context should be used
        use_context = False
        input_context_df = None

        if os.path.exists(additional_context_path):
            print(Fore.YELLOW + "Additional context file found.")
            use_context_input = input(Fore.YELLOW + "Do you want to use the context? (Yes/No): ").strip().lower()
            
            if use_context_input in ["yes", "y"]:
                use_context = True
                print(Fore.CYAN + "Loading additional context...")
                input_context_df = load_dataframe(additional_context_path)
                print(Fore.GREEN + f"✓ Loaded {len(input_context_df)} context records")
                print()

        # Generate manual tests
        if use_context and input_context_df is not None:
            processor.gen_manual_test_context(
                input_us_df=input_us_df,
                test_type=manual_test_type,
                input_context_df=input_context_df,
                output_dir=output_file_path
            )
        else:
            processor.generate_manual_test(
                input_us_df=input_us_df,
                test_type=manual_test_type,
                output_dir=output_file_path
            )

        print()
        print(Fore.GREEN + Style.BRIGHT + "✓ Manual test generation completed successfully!")
        print(Fore.YELLOW + f"Output saved to: {output_file_path}")
        print()

    except FileNotFoundError as e:
        print(Fore.RED + Style.BRIGHT + f"Error: {str(e)}")
        print(Fore.YELLOW + "Please ensure the input file exists and ConfigIO.properties is configured correctly.")
        sys.exit(1)

    except ValueError as e:
        print(Fore.RED + Style.BRIGHT + f"Error: {str(e)}")
        print(Fore.YELLOW + "Please check your environment variables and configuration files.")
        print(Fore.YELLOW + "See .env.example for required Azure OpenAI settings.")
        sys.exit(1)

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
