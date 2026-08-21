import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Steel Fatigue Predictor", page_icon="⚙️", layout="wide")
st.title("Steel Fatigue Strength Predictor & Optimizer")

# Domain of Applicability Update
st.info("ℹ️ **Domain of Applicability:** This predictive model is trained strictly on the NIMS dataset. It is accurate for **carbon and low-alloy steels, carburizing steels, and spring steels**. Input boundaries are locked to the exact training distribution to prevent out-of-distribution (OOD) AI hallucinations.")

# --- 2. LOAD THE MODEL ---
# Using @st.cache_resource so that the model is loaded only once and cached for future use to avoid slowing down the site.
@st.cache_resource
def load_model():
    return joblib.load('steel_fatigue_ann_model.pkl')

try:
    model = load_model()
    st.success("Artificial Neural Network Model Loaded Successfully!!")
except Exception as e:
    st.error(f"Couldn't load model!!.⚠️  Error: {e}")

# --- 3. CREATING TABS FOR UI ---
tab1, tab2 = st.tabs(["Manual Predictor (Forward)", "Smart GA Optimizer (Inverse)"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("Predict Strength from Chemical Composition")
    st.markdown("Enter the exact chemical composition and heat treatment parameters below:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Heat Treatment (Phase 1)**")
        f1 = st.number_input("1. NT (°C)", min_value=800.0, max_value=950.0, value=870.0)
        f2 = st.number_input("2. THT (°C)", min_value=800.0, max_value=950.0, value=870.0)
        f3 = st.number_input("3. THt (h)", min_value=0.0, max_value=5.0, value=1.0)
        f4 = st.number_input("4. THQCr (°C/s)", min_value=5.0, max_value=30.0, value=20.0)
        f5 = st.number_input("5. CT (°C)", min_value=0.0, max_value=20.0, value=0.0)
        f6 = st.number_input("6. Ct (h)", min_value=0.0, max_value=5.0, value=0.0)
        f7 = st.number_input("7. DT (°C)", min_value=0.0, max_value=20.0, value=0.0)
        
    with col2:
        st.markdown("**Heat Treatment (Phase 2)**")
        f8 = st.number_input("8. Dt (h)", min_value=0.0, max_value=5.0, value=0.0)
        f9 = st.number_input("9. QmT (°C)", min_value=0.0, max_value=50.0, value=20.0)
        f10 = st.number_input("10. TT (°C)", min_value=400.0, max_value=650.0, value=500.0)
        f11 = st.number_input("11. Tt (h)", min_value=0.5, max_value=3.0, value=1.0)
        f12 = st.number_input("12. TCr (°C/s)", min_value=1.0, max_value=20.0, value=10.0)
        st.markdown("**Chemical Comp. 1**")
        f13 = st.number_input("13. Carbon (C) %", min_value=0.1, max_value=0.6, value=0.30)
        f14 = st.number_input("14. Silicon (Si) %", min_value=0.1, max_value=1.5, value=0.20)
        
    with col3:
        st.markdown("**Chemical Comp. 2**")
        f15 = st.number_input("15. Manganese (Mn) %", min_value=0.4, max_value=2.0, value=0.80)
        f16 = st.number_input("16. Phosphorus (P) %", min_value=0.005, max_value=0.03, value=0.01, format="%0.3f")
        f17 = st.number_input("17. Sulfur (S) %", min_value=0.005, max_value=0.03, value=0.01, format="%0.3f")
        f18 = st.number_input("18. Nickel (Ni) %", min_value=0.0, max_value=3.0, value=0.00)
        f19 = st.number_input("19. Chromium (Cr) %", min_value=0.0, max_value=2.5, value=1.00)
        f20 = st.number_input("20. Copper (Cu) %", min_value=0.0, max_value=1.0, value=0.00)
        f21 = st.number_input("21. Molybdenum (Mo) %", min_value=0.0, max_value=1.0, value=0.20)
        
    with col4:
        st.markdown("**Mechanical / Inclusions**")
        f22 = st.number_input("22. RedRatio", min_value=300.0, max_value=800.0, value=500.0)
        f23 = st.number_input("23. dA", min_value=0.0, max_value=10.0, value=0.0)
        f24 = st.number_input("24. dB", min_value=0.0, max_value=10.0, value=0.0)
        f25 = st.number_input("25. dC", min_value=0.0, max_value=10.0, value=0.0)
        
    st.markdown("---") # Just to separate the UI with a horizontal line
    
    # Prediction Button
    if st.button("Predict Fatigue Strength ", use_container_width=True):
        
        # Packing the 25 variables into a 2D array because sklearn models require 2D arrays as input
        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, 
                                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, 
                                f21, f22, f23, f24, f25]])
        
        # Asking for predictions from the loaded model which is ANN (Artificial Neural Network)
        prediction = model.predict(input_data)[0]
        
        # Printing the results
        st.success(f"### Predicted Fatigue Strength: {prediction:.2f} MPa")
        st.balloons() # Just a fun animation upon calculation of result :)


# --- TAB 2: OPTIMIZER ---
with tab2:
    st.header("Find Best Composition for Target Strength (Inverse Design)")
    st.markdown("Specify your required strength. The AI will reverse-engineer the exact 25-parameter composition!")
    
    # Info box explaining the GA stochastic behavior
    st.info("💡 **Did you know?** You might get different recipes for the same target strength! This isn't a bug—it mirrors real-world metallurgy. Just like a factory can achieve a specific strength using various combinations of heat treatments and alloys, our AI dynamically explores millions of valid combinations to find the fastest viable recipe for you.")
    
    # User target input
    target_mpa = st.number_input("Target Fatigue Strength (MPa)", min_value=200.0, max_value=1200.0, value=650.0, step=10.0)
    
    st.markdown("---")
    st.subheader("Constraints & Fixed Parameters")
    st.info("Check the box to lock a parameter and set its fixed value. AI will optimize the rest.")
    
    # 1. List of all the 25 features , according to their order in the actual dataset, with their (Name, Min, Max, Default)
    features = [
        ("NT (°C)", 800.0, 950.0, 870.0),      # 0
        ("THT (°C)", 800.0, 950.0, 870.0),     # 1
        ("THt (h)", 0.0, 5.0, 1.0),            # 2
        ("THQCr (°C/s)", 5.0, 30.0, 20.0),     # 3
        ("CT (°C)", 0.0, 20.0, 0.0),           # 4
        ("Ct (h)", 0.0, 5.0, 0.0),             # 5
        ("DT (°C)", 0.0, 20.0, 0.0),           # 6
        ("Dt (h)", 0.0, 5.0, 0.0),             # 7
        ("QmT (°C)", 0.0, 50.0, 20.0),         # 8
        ("TT (°C)", 400.0, 650.0, 500.0),      # 9
        ("Tt (h)", 0.5, 3.0, 1.0),             # 10
        ("TCr (°C/s)", 1.0, 20.0, 10.0),       # 11
        ("Carbon (C) %", 0.1, 0.6, 0.30),      # 12
        ("Silicon (Si) %", 0.1, 1.5, 0.20),    # 13
        ("Manganese (Mn) %", 0.4, 2.0, 0.80),  # 14
        ("Phosphorus (P) %", 0.005, 0.03, 0.01),# 15
        ("Sulfur (S) %", 0.005, 0.03, 0.01),   # 16
        ("Nickel (Ni) %", 0.0, 3.0, 0.0),      # 17
        ("Chromium (Cr) %", 0.0, 2.5, 1.0),    # 18
        ("Copper (Cu) %", 0.0, 1.0, 0.0),      # 19
        ("Molybdenum (Mo) %", 0.0, 1.0, 0.20), # 20
        ("RedRatio", 300.0, 800.0, 500.0),     # 21
        ("dA", 0.0, 10.0, 0.0),                # 22
        ("dB", 0.0, 10.0, 0.0),                # 23
        ("dC", 0.0, 10.0, 0.0)                 # 24
    ]

    fixed_values_dict = {}  # Dictionary to remember which feature the user locked
    bounds = []             # List to feed into GA
    
    # 2. Dynamic UI for all 25 constraints inside an expander
    with st.expander("⚙️ Click to Expand & Set Constraints for all 25 Parameters", expanded=False):
        cols = st.columns(4)
        
        for i, (name, min_v, max_v, def_v) in enumerate(features):
            col = cols[i % 4]
            with col:
                st.markdown(f"**{name}**")
                is_locked = st.checkbox(f"Lock {name}", key=f"lock_{i}")
                if is_locked:
                    val = st.number_input(f"Set Value", min_value=min_v, max_value=max_v, value=def_v, key=f"val_{i}")
                    fixed_values_dict[i] = val
                    # Lock the boundaries very tightly for GA
                    bounds.append([val, val + 1e-5]) 
                else:
                    # Keep original broad boundaries
                    bounds.append([min_v, max_v])
                st.markdown("---") # Seperating into parts
                
    st.write("") # Adding space
    
    if st.button("Run AI Optimizer ", use_container_width=True):
        
        # UI Spinner while calculating
        with st.spinner("AI is evaluating millions of combinations. Please wait ~40 seconds..."):
            
            # Step 3: Fitness Function
            def fitness_function(composition):
                comp_2d = np.array(composition).reshape(1, -1)
                predicted_mpa = model.predict(comp_2d)[0]
                error = abs(target_mpa - predicted_mpa)
                return error

            # Step 4: Run the Engine (Increased Iterations for higher accuracy)
            result = differential_evolution(
                fitness_function, 
                bounds, 
                maxiter=100,   
                popsize=20,    
                tol=0.01       # Strict tolerance
            )
            
            # Extract Results
            best_recipe = result.x
            achieved_strength = model.predict(best_recipe.reshape(1, -1))[0]
            
            st.success(f"### Optimization Complete! Achieved Strength: {achieved_strength:.2f} MPa")
            st.balloons()
            
            st.markdown("#### Complete Optimized Composition (All 25 Parameters):")
            
            # 5. Display ALL results in a clean 4-column layout
            res_cols = st.columns(4)
            for i, (name, _, _, _) in enumerate(features):
                res_col = res_cols[i % 4]
                with res_col:
                    # If the user had locked, then it'll show 🔒 icon , otherwise 🟢
                    if i in fixed_values_dict:
                        st.info(f"🔒 **{name}:**\n{best_recipe[i]:.3f}")
                    else:
                        st.success(f"🟢 **{name}:**\n{best_recipe[i]:.3f}")
