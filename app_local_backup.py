import streamlit as st
from provenance_decoder import analyze_failure_mode, log_to_google_sheet

def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="ASL Sovereign Decoder Ring", 
        page_icon="🔍", 
        layout="centered"
    )
    
    # 2. Header and Context
    st.title("The Provenance Decoder Ring 🔍")
    st.markdown("""
    **Stop going to the hardware store for bread.** Tell us what you're experiencing and we can diagnose the issue and save you time, resources, and bandwidth for what really matters.  
    """)
    
    st.divider()

    # 3. User Input
    st.subheader("Input Error Log or User Complaint")
    user_text = st.text_area(
        "Drop the frustration here:", 
        placeholder="e.g., 'it misses my face and I have to perform for the camera just to get it to see me'",
        height=120
    )

    # 4. Action Button & Execution
    if st.button("Diagnose Model Provenance", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a complaint to diagnose.")
        else:
            with st.spinner("Gemma 4 is analyzing the failure geometry..."):
                results = analyze_failure_mode(user_text, model_name="asl_project")
                
                # --- ADD THIS LINE TO SAVE TO GOOGLE SHEETS ---
                log_to_google_sheet(user_text, results)
                
            st.success("Diagnosis Complete and Logged to Cloud.")
            
            # 5. Result Rendering
            st.subheader("Diagnostic Fingerprint")
            
            # Use columns for a clean dashboard look
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Detected Failure Mode", value=results.get("failure_mode", "Error"))
            with col2:
                st.metric(label="Compute Optimization", value=results.get("compute_saved", "N/A"))
                
            # Distinct colored boxes for the primary takeaways
            st.info(f"**Training Data Fingerprint:**\n\n{results.get('provenance_diagnosis', 'No diagnosis returned.')}")
            st.error(f"**Actionable Recommendation:**\n\n{results.get('recommendation', 'No recommendation returned.')}")

            # Optional: Show raw output if it failed to parse correctly
            if "raw_output" in results:
                with st.expander("View Raw Model Output"):
                    st.write(results["raw_output"])

    st.divider()
    st.caption("Powered by Gemma  (Local/Ollama) | Framework: *FSboard confusion matrix diagnostic decoder ring* | Veracity improves with use )

if __name__ == "__main__":
    main()