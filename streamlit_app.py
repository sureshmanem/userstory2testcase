"""
streamlit_app.py
Streamlit UI for manual test case generation.
"""

import streamlit as st
import pandas as pd
import io
from us_to_mtc_file.ui_runner import generate_manual_tests_dataframe, UIRunOptions


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load DataFrame from uploaded file with encoding fallback.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Loaded DataFrame.
    """
    if uploaded_file.name.endswith('.csv'):
        # Try multiple encodings for CSV
        for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                return pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Failed to decode CSV file with any encoding")
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload .csv or .xlsx file")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="User Story to Manual Test Case Generator",
        page_icon="📝",
        layout="wide"
    )

    st.title("📝 User Story to Manual Test Case Generator")
    st.markdown("Generate manual test cases from user stories using Azure OpenAI")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # File uploaders
    st.sidebar.subheader("📂 Input Files")
    user_stories_file = st.sidebar.file_uploader(
        "User Stories (Required)",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing user stories with Description and AcceptanceCriteria columns"
    )

    additional_context_file = st.sidebar.file_uploader(
        "Additional Context (Optional)",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing additional context for retrieval"
    )

    # Options
    st.sidebar.subheader("🎛️ Generation Options")

    test_type = st.sidebar.selectbox(
        "Test Type",
        ["Both", "Positive", "Negative"],
        help="Type of manual tests to generate"
    )

    use_context = st.sidebar.checkbox(
        "Use Additional Context",
        value=False,
        help="Enable context retrieval from uploaded additional context file",
        disabled=(additional_context_file is None)
    )

    num_context_retrieve = st.sidebar.number_input(
        "# Contexts to Retrieve",
        min_value=1,
        max_value=20,
        value=3,
        help="Number of context chunks to retrieve per user story",
        disabled=(not use_context)
    )

    enable_additional_intelligence = st.sidebar.checkbox(
        "Enable Additional Intelligence",
        value=True,
        help="Allow LLM to use its own knowledge in addition to provided context"
    )

    generate_additional_ac = st.sidebar.checkbox(
        "Generate Additional Acceptance Criteria",
        value=False,
        help="Generate additional acceptance criteria before creating manual tests"
    )

    # Main content
    if user_stories_file is None:
        st.info("👈 Please upload a User Stories file to get started")
        st.markdown("""
        ### Expected Format
        
        Your user stories file should contain at least these columns:
        - **Description**: User story description
        - **AcceptanceCriteria**: Acceptance criteria for the story
        - **ID** or **FormattedID** or **UserStoryID**: (Optional) Unique identifier
        
        ### Azure OpenAI Setup
        
        Make sure you have configured your Azure OpenAI credentials in a `.env` file:
        ```
        AZURE_OPENAI_API_KEY=your_api_key
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        AZURE_OPENAI_API_VERSION=2024-02-15-preview
        AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
        ```
        """)
        return

    # Load user stories
    try:
        user_stories_df = load_uploaded_file(user_stories_file)
        st.success(f"✅ Loaded {len(user_stories_df)} user stories")

        # Show preview
        with st.expander("Preview User Stories"):
            st.dataframe(user_stories_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading user stories file: {str(e)}")
        return

    # Load additional context if provided
    additional_context_df = None
    if additional_context_file is not None:
        try:
            additional_context_df = load_uploaded_file(additional_context_file)
            st.success(f"✅ Loaded {len(additional_context_df)} context records")

            with st.expander("Preview Additional Context"):
                st.dataframe(additional_context_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading additional context file: {str(e)}")
            return

    # Generate button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🚀 Generate Manual Test Cases",
            type="primary",
            use_container_width=True
        )

    if generate_button:
        # Prepare options
        options = UIRunOptions(
            test_type=test_type,
            use_context=use_context and additional_context_df is not None,
            num_context_retrieve=num_context_retrieve,
            enable_additional_intelligence=enable_additional_intelligence,
            generate_additional_acceptance_criteria=generate_additional_ac
        )

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_log = st.empty()

        def progress_callback(current, total, message):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            # Generate manual tests
            results_df = generate_manual_tests_dataframe(
                user_stories_df=user_stories_df,
                additional_context_df=additional_context_df,
                options=options,
                progress_callback=progress_callback
            )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            st.success("✅ Manual test cases generated successfully!")

            # Display results
            st.markdown("---")
            st.subheader("📊 Results")

            # Metrics summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("User Stories Processed", len(results_df))
            with col2:
                st.metric("Total LLM Calls", results_df["LLMCalls"].sum())
            with col3:
                st.metric("Total Tokens", f"{results_df['TotalTokens'].sum():,}")
            with col4:
                st.metric("Avg Latency (s)", f"{results_df['LatencySeconds'].mean():.2f}")

            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Used Context", results_df[results_df["UsedContext"] == "Yes"].shape[0])
            with col2:
                st.metric("Avg Tokens/Second", f"{results_df['TokensPerSecond'].mean():.2f}")
            with col3:
                total_effort_saved = results_df["EstimatedEffortSavedMinutes"].sum()
                st.metric("Est. Effort Saved", f"{total_effort_saved:.1f} min")

            # Results table (without prompt and metrics for display)
            st.markdown("### Generated Manual Tests")
            display_columns = ["UserStoryID", "TestType", "UsedContext", "UserStory", "Context", "ManualTest"]
            display_df = results_df[display_columns]
            st.dataframe(display_df, use_container_width=True, height=400)

            # Expandable sections for each story
            st.markdown("### Detailed Results")
            for idx, row in results_df.iterrows():
                with st.expander(f"Story {row['UserStoryID']} - Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**User Story:**")
                        st.text_area(
                            "User Story",
                            row["UserStory"],
                            height=150,
                            key=f"us_{idx}",
                            label_visibility="collapsed"
                        )
                        
                        if row["Context"]:
                            st.markdown("**Retrieved Context:**")
                            st.text_area(
                                "Context",
                                row["Context"],
                                height=150,
                                key=f"ctx_{idx}",
                                label_visibility="collapsed"
                            )
                    
                    with col2:
                        st.markdown("**Generated Manual Test:**")
                        st.text_area(
                            "Manual Test",
                            row["ManualTest"],
                            height=150,
                            key=f"mt_{idx}",
                            label_visibility="collapsed"
                        )
                        
                        st.markdown("**Metrics:**")
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.text(f"LLM Calls: {row['LLMCalls']}")
                            st.text(f"Total Tokens: {row['TotalTokens']}")
                            st.text(f"Latency: {row['LatencySeconds']}s")
                        with metrics_col2:
                            st.text(f"Tokens/Sec: {row['TokensPerSecond']}")
                            st.text(f"Output Words: {row['OutputWords']}")
                            st.text(f"Effort Saved: {row['EstimatedEffortSavedMinutes']:.1f}m")
                    
                    st.markdown("**Prompt Sent to LLM:**")
                    st.code(row["PromptSentToLLM"], language="text")

            # Download button
            st.markdown("---")
            st.subheader("💾 Download Results")

            # Convert to CSV
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="manual_test_cases.csv",
                mime="text/csv",
                use_container_width=True
            )

        except RuntimeError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ {str(e)}")
            st.info("💡 Please check your .env file and ensure all Azure OpenAI credentials are configured correctly.")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
