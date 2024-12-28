

##### Documentation Block 

# class DSIAnalysisApp:
    # """Main application class for DSI Research Abstract Analysis"""

    # # Initialization Methods
    # def __init__(self):
        # """Initialize application components"""
        # Use case: App startup and component initialization
        # Called by: main() function when creating app instance
        # Dependencies: ExcelHandler

    # # File Handling Methods
    # def save_uploaded_file(self, uploaded_file):
        # """Handle file uploads"""
        # Use case: Save temporary files from Streamlit uploads
        # Called by: run() method during file processing
        # Returns: Path to saved file or None

    # # Validation Methods
    # def validate_api_key(self, api_key: str) -> bool:
        # """Validate API keys"""
        # Use case: Verify OpenAI API key validity
        # Called by: run() method before processing
        # Returns: Boolean validation result

    # # Core Processing Methods
    # def process_abstracts(self, df_abstracts: pd.DataFrame) -> pd.DataFrame:
        # """Process abstracts and generate analysis"""
        # Use case: Main analysis of research abstracts
        # Called by: run() method when "Start Analysis" clicked
        # Dependencies: AbstractAnalyzer
        # Returns: DataFrame with analysis results

    # # Output Handling Methods
    # def save_and_download_results(self, df_results):
        # """Save and provide download for results"""
        # Use case: Create downloadable output file
        # Called by: run() method after processing
        # Returns: Output filename or None

    # # Main Application Flow
    # def run(self):
        # """Main application execution"""
        # Use case: Control main application flow and UI
        # Called by: main() function
        # Dependencies: All other methods

# Method Call Flow:
# 1. main() → DSIAnalysisApp() 
# 2. main() → app.run()
# 3. run() → validate_api_key()
# 4. run() → save_uploaded_file()
# 5. run() → process_abstracts()
# 6. run() → save_and_download_results()








import streamlit as st
import pandas as pd
import os
import tempfile
import time
import tempfile
import shutil
import math
import time

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from google.generativeai import GenerativeModel
from typing import Dict

from excel_handler import ExcelHandler
from abstract_analyzer import AbstractAnalyzer

from datetime import datetime




class DSIAnalysisApp:
    
    
    def __init__(self):
        self.excel_handler = ExcelHandler()
        self.analyzer = None  # Initialize as None
        self.dsi_structure = None
        
    
    
    def get_model_options(self):
       return {
       "OpenAI Models": {
           "GPT-4": {"name": "gpt-4", "pricing": {"input": 0.03, "output": 0.06}, "cost_1M": {"total": "$90"}, "max_tokens": 8192, "params": "1.76T", "api_type": "OpenAI"},
           "GPT-4 Turbo": {"name": "gpt-4-turbo-preview", "pricing": {"input": 0.01, "output": 0.03}, "cost_1M": {"total": "$40"}, "max_tokens": 128000, "params": "1.76T", "api_type": "OpenAI"},
           "GPT-3.5 Turbo": {"name": "gpt-3.5-turbo", "pricing": {"input": 0.0005, "output": 0.0015}, "cost_1M": {"total": "$2"}, "max_tokens": 16384, "params": "175B", "api_type": "OpenAI"}
       },
       "Anthropic Models": {
           "Claude 3 Opus": {"name": "claude-3-opus-20240229", "pricing": {"input": 0.03, "output": 0.09}, "cost_1M": {"total": "$120"}, "max_tokens": 200000, "params": "2.7T", "api_type": "Anthropic"},
           "Claude 3 Sonnet": {"name": "claude-3-5-sonnet-20241022", "pricing": {"input": 0.015, "output": 0.045}, "cost_1M": {"total": "$60"}, "max_tokens": 200000, "params": "1.3T", "api_type": "Anthropic"}
       },
      
        "Google Models": {
            "Gemini-1.5-Pro": {
                "name": "gemini-1.5-pro",  # Update model name
                "pricing": {"input": 0.00025, "output": 0.0005},
                "cost_1M": {"total": "$0.75"},
                "max_tokens": 32000,
                "params": "450B",
                "api_type": "Google"
            },
            "Gemini-1.5-Pro-Latest": {
                "name": "gemini-1.5-pro-latest",  # For latest version
                "pricing": {"input": 0.00025, "output": 0.0005},
                "cost_1M": {"total": "$0.75"},
                "max_tokens": 32000,
                "params": "450B",
                "api_type": "Google"
            }
        
        },
      
       "DeepInfra Models": {
           
           "Mixtral-8x7B": {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "pricing": {"input": 0.24, "output": 0.24}, "cost_1M": {"total": "$0.48"}, "max_tokens": 32768, "params": "47B", "api_type": "DeepInfra"},
           "Llama-3-70B": {"name": "meta-llama/Llama-3.3-70B-Instruct", "pricing": {"input": 0.30, "output": 0.45}, "cost_1M": {"total": "$0.75"}, "max_tokens": 4096, "params": "70B", "api_type": "DeepInfra"},
           "Laama-3 70 Turbo": {"name": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "pricing": {"input": 0.20, "output": 0.20}, "cost_1M": {"total": "$0.40"}, "max_tokens": 4096, "params": "34B", "api_type": "DeepInfra"},
           "Llama-2-13B": {"name": "llama-2-13b", "pricing": {"input": 0.15, "output": 0.25}, "cost_1M": {"total": "$0.40"}, "max_tokens": 4096, "params": "13B", "api_type": "DeepInfra"}
       },
       "Together.ai Models": {
           "Mixtral-8x7B": {"name": "mixtral-8x7b-instruct", "pricing": {"input": 0.20, "output": 0.70}, "cost_1M": {"total": "$0.90"}, "max_tokens": 32768, "params": "47B", "api_type": "Together"},
           "Llama-2-70B": {"name": "llama-2-70b-chat", "pricing": {"input": 0.70, "output": 0.90}, "cost_1M": {"total": "$1.60"}, "max_tokens": 4096, "params": "70B", "api_type": "Together"}
       },
       "Groq Models": {
           "Mixtral-8x7B": {"name": "mixtral-8x7b-32768", "pricing": {"input": 0.24, "output": 0.24}, "cost_1M": {"total": "$0.48"}, "max_tokens": 32768, "params": "47B", "api_type": "Groq"},
           "Llama-3-70B": {"name": "Meta-Llama-3-70B-Instruct", "pricing": {"input": 0.59, "output": 0.79}, "cost_1M": {"total": "$1.38"}, "max_tokens": 4096, "params": "70B", "api_type": "Groq"},
           "Llama-3-13B": {"name": "llama-3.3-70b-versatile","pricing" : {"input": 0.20, "output": 0.30}, "cost_1M": {"total": "$0.50"}, "max_tokens": 4096, "params": "13B", "api_type": "Groq"}
       },
       "X.AI Models": {
            "Grok-Beta": {"name": "grok-beta", "pricing": {"input": 0.0, "output": 0.0}, "cost_1M": {"total": "$0"}, "max_tokens": 8192, "params": "N/A", "api_type": "XAI"}
                }
   
   
   
   
   }
    


    
   
    
    
    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file and return path"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return None

    
    
    
    def validate_api_key(self, api_key: str, model_info: dict) -> bool:
        try:
            api_type = model_info['api_type']
            
            if api_type == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_info['name'],
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.success("✅ OpenAI API key validated successfully")
                
            elif api_type == "Anthropic":
                client = Anthropic(api_key=api_key)
                message = client.messages.create(
                    model=model_info['name'],
                    messages=[{
                        "role": "user",
                        "content": "test"
                    }],
                    max_tokens=5
                )
                st.success("✅ Anthropic API key validated successfully")
                return True
            
            elif api_type == "Google":
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_info['name'])
                    response = model.generate_content("test")
                    if not response or not response.text:
                        raise ValueError("No valid response from Google API")
                    st.success("✅ Google API key validated successfully")
                    return True
                except Exception as e:
                    st.error(f"❌ Google API Error: {str(e)}")
                    return False
            
            elif api_type == "DeepInfra":
                client = OpenAI(
                    api_key=api_key, 
                    base_url="https://api.deepinfra.com/v1/openai"
                )
                response = client.chat.completions.create(
                    model=model_info['name'],
                    messages=[{
                        "role": "user",
                        "content": "Hello"
                    }],
                    max_tokens=10
                )
                if response:
                    st.success("✅ DeepInfra API key valid")
                    return True
                st.error("❌ DeepInfra API validation failed")
                return False

            elif api_type == "XAI":
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1"
                )
                response = client.chat.completions.create(
                    model=model_info['name'],
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    st.success("✅ X.AI API key valid")
                    return True
                st.error("❌ Invalid X.AI response")
                return False
  
            
            elif api_type == "Groq":
                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1",
                        default_headers={"groq-api-version": "2024-03-25"}
                    )
                    response = client.chat.completions.create(
                        model=model_info['name'],
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=10,
                        temperature=1e-8
                    )
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        st.success("✅ Groq API key valid")
                        return True
                    st.error("❌ Invalid Groq response")
                    return False
                except Exception as e:
                    st.error(f"❌ Groq API Error: {str(e)}")
                    return False
          





        except Exception as e:
            error_messages = {
                "OpenAI": "Invalid OpenAI API key or rate limit exceeded",
                "Anthropic": "Invalid Anthropic API key or rate limit exceeded", 
                "Google": "Invalid Google API key or service error"
            }
            st.error(f"❌ {error_messages.get(api_type, 'API validation failed')}: {str(e)}")
            return False

   
    
    
    def analyze_token_usage(self, text: str, tag: str = "") -> Dict[str, int]:
        TOKEN_ESTIMATES = {
            'abstract': 1.3,
            'theme_desc': 2.0,
            'json_output': 1.5
        }

    
    def count_component_tokens(content: str, multiplier: float) -> float:
        return len(str(content).split()) * multiplier 

        token_breakdown = {
        'abstract': count_component_tokens(text, TOKEN_ESTIMATES['abstract']),
        't00_summary': 1000,
        'theme_descriptions': 1000, 
        't00_output': 300,
        'theme_matching': 50,
        'evidence_quotes': 200
        }

        return token_breakdown
    

    def process_abstract_batches(self, df_abstracts: pd.DataFrame) -> pd.DataFrame:
        BATCH_SIZE = 5
        BASE_DELAY = 45  
        MAX_RETRIES = 3
        BACKOFF_MULTIPLIER = 2
        TOKEN_THRESHOLD = 12000

        results_dfs = []
        total_batches = math.ceil(len(df_abstracts)/BATCH_SIZE)
        
        st.info("⏳ Starting batch processing...")
        progress_bar = st.progress(0)
        
        batch_metrics = {'total_tokens': 0, 'total_time': 0}

        for i in range(0, len(df_abstracts), BATCH_SIZE):
            batch_num = i//BATCH_SIZE + 1
            batch = df_abstracts[i:i + BATCH_SIZE].copy()  # Create copy of batch
            
            batch_start = time.time()
            st.write(f"📊 Processing batch {batch_num} of {total_batches}")

            batch_tokens = 0
            for idx, row in batch.iterrows():
                if pd.notna(row['Abstract']):
                    tokens = self.analyze_token_usage(str(row['Abstract']))
                    batch_tokens += sum(tokens.values()) if isinstance(tokens, dict) else 0

            try:
                results = self.process_abstracts(batch)
                if not results.empty:
                    results_dfs.append(results)
                    
                    batch_time = time.time() - batch_start
                    batch_metrics['total_tokens'] += batch_tokens
                    batch_metrics['total_time'] += batch_time
                    
                    progress = (batch_num/total_batches)
                    progress_bar.progress(progress)

            except Exception as e:
                st.error(f"Batch {batch_num} error: {str(e)}")
                error_results = batch.copy()
                error_results['Type'] = f"Error: {str(e)}"
                results_dfs.append(error_results)

        if not results_dfs:
            raise Exception("No results generated")
            
        return pd.concat(results_dfs, ignore_index=True)
        
    
   

        
        
        
        ##################################
    def process_abstracts(self, df_abstracts: pd.DataFrame) -> pd.DataFrame:
        """
        Process abstracts and generate analysis results with preserved column structure
        
        Args:
            df_abstracts (pd.DataFrame): Input DataFrame containing abstracts with columns:
                - AbstractID: Unique identifier
                - Abstract: Text content
                [Original columns preserved]
                
        Returns:
            pd.DataFrame: Results DataFrame with columns:
                [Original columns preserved]
                Added analysis columns:
                - Type: Research type classification
                - Methods: Research methods used
                - Digital Elements: Digital technologies identified
                - Social Elements: Social aspects identified  
                - Key Findings: Main findings extracted
                - SDG01-17 Match: SDG alignment (Yes/No)
                - SDG01-17 Evidence: Evidence for SDG alignment
                - T01-T10 Match: Theme matches (Yes/No) 
                - T01-T10 Evidence: Evidence for theme matches
        """
        # Store original columns to preserve
        original_columns = df_abstracts.columns.tolist()
        
        # Define analysis result columns
        analysis_columns = [
            "Type", "Methods", "Digital Elements", "Social Elements", 
            "Key Findings"
        ]
        
        # Define SDG columns (01-17)
        sdg_columns = []
        for i in range(1, 18):
            sdg_columns.extend([
                f"SDG{i:02d} Match",
                f"SDG{i:02d} Evidence"
            ])
            
        # Define Theme columns (T01-T10) 
        theme_columns = []
        for i in range(1, 11):
            theme_columns.extend([
                f"T{i:02d} Match",
                f"T{i:02d} Evidence"
            ])
        
        # Initialize results list
        results_list = []
        
        # Process each abstract
        for idx in range(len(df_abstracts)):
            if pd.notna(df_abstracts.iloc[idx]['Abstract']):
                try:
                    # Get analysis results
                    results = self.analyzer.analyze_abstract(
                        str(df_abstracts.iloc[idx]['Abstract']),
                        abstract_number=idx + 1
                    )
                    
                    # Initialize result row with original data
                    result_row = df_abstracts.iloc[idx].to_dict()
                    
                    # Add analysis results
                    result_row.update({
                        "Type": results.get("T00", {}).get("Type", "NA"),
                        "Methods": results.get("T00", {}).get("Methods", "NA"),
                        "Digital Elements": results.get("T00", {}).get("Digital_Elements", "NA"),
                        "Social Elements": results.get("T00", {}).get("Social_Elements", "NA"),
                        "Key Findings": results.get("T00", {}).get("Key_Findings", "NA")
                    })

                    # Add SDG results
                    sdg_alignments = results.get("T00", {}).get("SDG_Alignments", {})
                    for i in range(1, 18):
                        sdg_code = f"SDG{i:02d}"
                        sdg_data = sdg_alignments.get(sdg_code, {"Match": "No", "Evidence": "NA"})
                        result_row[f"{sdg_code} Match"] = sdg_data["Match"]
                        result_row[f"{sdg_code} Evidence"] = sdg_data["Evidence"]

                    # Add Theme results
                    for i in range(1, 11):
                        theme_id = f"T{i:02d}"
                        theme_data = results.get(theme_id, {})
                        result_row[f"{theme_id} Match"] = theme_data.get("Match", "No")
                        evidence = theme_data.get("Evidence", {})
                        content = evidence.get("Matched_Content", [])
                        result_row[f"{theme_id} Evidence"] = " || ".join(content) if content else "NA"

                    results_list.append(result_row)

                except Exception as e:
                    print(f"Error processing abstract {idx + 1}: {e}")
                    # Preserve original data on error
                    error_row = df_abstracts.iloc[idx].to_dict()
                    error_row.update({col: "NA" for col in analysis_columns + sdg_columns + theme_columns})
                    error_row["Type"] = f"Error: {str(e)}"
                    results_list.append(error_row)

        # Create DataFrame with all columns
        df_results = pd.DataFrame(results_list)
        
        # Ensure all expected columns exist
        all_columns = original_columns + analysis_columns + sdg_columns + theme_columns
        for col in all_columns:
            if col not in df_results.columns:
                df_results[col] = "NA"
        
        # Order columns correctly
        df_results = df_results[all_columns]
        
        return df_results



    def save_and_download_results(self, df_results, start_no, end_no, original_filename):
        """Save results and create download button with range in filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract original filename without extension
        base_filename = os.path.splitext(original_filename)[0]
        output_filename = f"ab{start_no:04d}_to_{end_no:04d}_{timestamp}_{base_filename}.xlsx"
        
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, output_filename)
            
            # Debug print
            print(f"Saving file to temp directory: {output_path}")
            
            # Save Excel file
            self.excel_handler.save_excel(df_results, output_path)
            
            # Create download button
            with open(output_path, 'rb') as f:
                file_data = f.read()
                st.download_button(
                    label="📥 Download Analysis Results",
                    data=file_data,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.success(f"✅ Analysis complete! Click above to download results.")
            st.write("Preview of analysis results:")
            st.dataframe(df_results.head())
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")
                
            return output_filename
                
        except Exception as e:
            st.error(f"Error saving results: {str(e)}")
            return None
        
        
        
        
    




    def run(self):
        """Main application execution with enhanced UI and error handling"""
        try:
            # Page configuration
            st.set_page_config(
                page_title="DSI Research Abstract Analysis",
                layout="wide",
                initial_sidebar_state="expanded"
            )

            # Main title
            st.markdown("""
            <h1 style='text-align: center; color: #1E88E5;'>
                DSI Research Abstract Analysis Tool
            </h1>
            """, unsafe_allow_html=True)

            # Initialize session state
            if 'selected_model' not in st.session_state:
                st.session_state.selected_model = None
            if 'model_info' not in st.session_state:
                st.session_state.model_info = None

            # Sidebar Configuration
            with st.sidebar:
                st.markdown("# DSI Analysis")
                st.markdown("---")
                
                # Model Selection Section
                st.header("🤖 Model Configuration")
                
                models = self.get_model_options()
                model_family = st.selectbox(
                    "Select AI Provider:",
                    options=list(models.keys()),
                    help="Choose the AI model provider for analysis"
                )
                
                specific_model = st.selectbox(
                    "Select Model:",
                    options=list(models[model_family].keys()),
                    help="Choose specific model version"
                )
                
                # Update session state
                st.session_state.model_info = models[model_family][specific_model]
                st.write("model info ", st.session_state.model_info)

                # Advanced Settings
                with st.expander("⚙️ Advanced Settings"):
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_info.get('default_temperature', 0.3),
                        help="Controls randomness in the output"
                    )
                    
                    batch_size = st.slider(
                        "Batch Size",
                        min_value=1,
                        max_value=50,
                        value=10,
                        help="Number of abstracts to process together"
                    )

                # Model Information Display
                st.markdown("### 📊 Model Details")
                st.info(f"""
                **Selected Model**: {st.session_state.model_info['name']}

                **Capabilities**:
                - Max Length: {st.session_state.model_info['max_tokens']:,} tokens
                - Parameters: {st.session_state.model_info['params']}

                **Pricing (per 1M tokens)**:
                - Input: ${st.session_state.model_info['pricing']['input']}
                - Output: ${st.session_state.model_info['pricing']['output']}
                - Total: {st.session_state.model_info['cost_1M']['total']}
                """)
            
            # Main Content Area
            main_col1, main_col2 = st.columns([2, 1])
            
            with main_col1:
                # API Key Input
                api_key = st.text_input(
                    f"🔑 Enter {st.session_state.model_info['api_type']} API Key:",
                    type="password",
                    help=f"Your {st.session_state.model_info['api_type']} API key is required for authentication"
                )

                #Add abstract range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_no = st.number_input("Starting Abstract Number", min_value=1, value=1)
                with col2:
                    end_no = st.number_input("Ending Abstract Number", min_value=1, value=10)

                # File Upload Section
                st.markdown("### 📁 File Upload")
                
                # File Format Instructions
                with st.expander("📋 File Format Requirements"):
                    st.markdown("""
                    #### DSI Master File Format:
                    - Excel file (.xlsx)
                    - Required columns:
                        1. ID (e.g., T01, T02, T03)
                        2. Theme Description
                        3. Matching Examples
                    
                    #### Abstracts File Format:
                    - Excel file (.xlsx)
                    - Required columns:
                        1. Abstract
                    """)

                # File uploaders
                dsi_master_file = st.file_uploader(
                    "Upload DSI Master File",
                    type=['xlsx'],
                    help="Upload your DSI master themes file"
                )
                
                abstracts_file = st.file_uploader(
                    "Upload Abstracts File",
                    type=['xlsx'],
                    help="Upload your research abstracts file"
                )

                # Process files if uploaded
                if dsi_master_file and abstracts_file and api_key:
                    if st.button("🚀 Start Analysis"):
                        with st.spinner("Validating API key..."):
                            validation_result = self.validate_api_key(api_key, st.session_state.model_info)
                            if not validation_result:
                                st.error(f"❌ API validation failed for {st.session_state.model_info['api_type']}")
                                return
                            
                            # Initialize analyzer here
                            st.write("Debug - model_info:", st.session_state.model_info)
                            self.analyzer = AbstractAnalyzer(
                                api_key=api_key,
                                model_info=st.session_state.model_info
                            )

                            master_path = self.save_uploaded_file(dsi_master_file)
                            abstracts_path = self.save_uploaded_file(abstracts_file)
                            
                            if master_path and abstracts_path:
                                try:
                                    # Load data using read_excel
                                    df_master = self.excel_handler.read_excel(master_path)
                                    df_abstracts = self.excel_handler.read_excel(abstracts_path)
                                    
                                    # Load DSI master data
                                    self.analyzer.load_dsi_master(master_path)
                                    
                                    # Process abstracts
                                    with st.spinner("Processing abstracts..."):
                                        results_df = self.process_abstract_batches(df_abstracts)
                                        #results_df = self.process_abstracts(df_abstracts)
                                    
                                    
                                   # Save and create download button with range info
                                    self.save_and_download_results(
                                        results_df, 
                                        start_no, 
                                        end_no,
                                        abstracts_file.name
                                        )
                            
                                    
                                    
                                except Exception as e:
                                    st.error(f"Error during analysis: {str(e)}")
                                    self.excel_handler.logger.error(f"Analysis error: {str(e)}")
                                    
                                finally:
                                    # Cleanup temporary files
                                    if master_path and os.path.exists(master_path):
                                        os.remove(master_path)
                                    if abstracts_path and os.path.exists(abstracts_path):
                                        os.remove(abstracts_path)
                            else:
                                st.error("Error saving uploaded files")
                else:
                    if not api_key:
                        st.warning("Please enter API key")
                    if not dsi_master_file:
                        st.warning("Please upload DSI master file")
                    if not abstracts_file:
                        st.warning("Please upload abstracts file")

            with main_col2:
                # Status and Information Panel
                st.markdown("### ℹ️ Status")
                st.info("""
                1. Upload your DSI master file
                2. Upload your abstracts file
                3. Enter your API key
                4. Click 'Start Analysis'
                """)
                
                # Progress Information
                if 'progress' in st.session_state:
                    st.progress(st.session_state.progress)
                    
                # Help and Support
                with st.expander("❓ Need Help?"):
                    st.markdown("""
                    For support:
                    - Check the documentation
                    - Contact technical support
                    - Visit our GitHub repository
                    """)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            if hasattr(self, 'excel_handler') and hasattr(self.excel_handler, 'logger'):
                self.excel_handler.logger.error(f"Application error: {str(e)}")
            st.exception(e)

        finally:
            # Cleanup code if needed
            if 'temp_files' in st.session_state:
                for file in st.session_state.temp_files:
                    if os.path.exists(file):
                        os.remove(file)
            
    
            
    
def main():
    app = DSIAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()