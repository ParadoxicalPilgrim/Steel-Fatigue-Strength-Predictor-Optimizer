import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Steel Fatigue Predictor", page_icon="⚙️", layout="wide")
st.title("⚙️ Steel Fatigue Strength Predictor & Optimizer")

# --- 2. LOAD THE MODEL ---
# Using @st.cache_resource so that the model is loaded only once and cached for future use to avoid slowing down the site.
@st.cache_resource
def load_model():
    return joblib.load('steel_fatigue_ann_model.pkl')

try:
    model = load_model()
    st.success("✅ Artificial Neural Network Model Loaded Successfully!")
except Exception as e:
    st.error(f"⚠️ Couldn't load model!!. Error: {e}")

# --- 3. CREATING TABS FOR UI ---
tab1, tab2 = st.tabs(["📊 Manual Predictor (Forward)", "🧠 Smart GA Optimizer (Inverse)"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("Predict Strength from Chemical Composition")
    st.markdown("Enter the exact chemical composition and heat treatment parameters below:")
    
    # 4 columns ka grid banayenge taaki UI clean aur professional lage
    col1, col2, col3, col4 = st.columns(4)
    
    # ⚠️ IMPORTANT: Yahan jo f1 se f25 tak variables hain, 
    # inka naam aur order tere Jupyter Notebook ke `X` dataframe ke columns jaisa hona chahiye!
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Heat Treatment (Phase 1)**")
        f1 = st.number_input("1. NT (°C)", value=870.0)
        f2 = st.number_input("2. THT (°C)", value=870.0)
        f3 = st.number_input("3. THt (h)", value=1.0)
        f4 = st.number_input("4. THQCr (°C/s)", value=20.0)
        f5 = st.number_input("5. CT (°C)", value=0.0)
        f6 = st.number_input("6. Ct (h)", value=0.0)
        f7 = st.number_input("7. DT (°C)", value=0.0)
        
    with col2:
        st.markdown("**Heat Treatment (Phase 2)**")
        f8 = st.number_input("8. Dt (h)", value=0.0)
        f9 = st.number_input("9. QmT (°C)", value=20.0)
        f10 = st.number_input("10. TT (°C)", value=500.0)
        f11 = st.number_input("11. Tt (h)", value=1.0)
        f12 = st.number_input("12. TCr (°C/s)", value=10.0)
        st.markdown("**Chemical Comp. 1**")
        f13 = st.number_input("13. Carbon (C) %", value=0.30)
        f14 = st.number_input("14. Silicon (Si) %", value=0.20)
        
    with col3:
        st.markdown("**Chemical Comp. 2**")
        f15 = st.number_input("15. Manganese (Mn) %", value=0.80)
        f16 = st.number_input("16. Phosphorus (P) %", value=0.01)
        f17 = st.number_input("17. Sulfur (S) %", value=0.01)
        f18 = st.number_input("18. Nickel (Ni) %", value=0.00)
        f19 = st.number_input("19. Chromium (Cr) %", value=1.00)
        f20 = st.number_input("20. Copper (Cu) %", value=0.00)
        f21 = st.number_input("21. Molybdenum (Mo) %", value=0.20)
        
    with col4:
        st.markdown("**Mechanical / Inclusions**")
        f22 = st.number_input("22. RedRatio", value=500.0)
        f23 = st.number_input("23. dA", value=0.0)
        f24 = st.number_input("24. dB", value=0.0)
        f25 = st.number_input("25. dC", value=0.0)
        
    st.markdown("---") # Ek horizontal line UI ko separate karne ke liye
    
    # Prediction Button
    if st.button("Predict Fatigue Strength 🚀", use_container_width=True):
        
        # In 25 variables ko ek single 2D array mein pack kar rahe hain
        # Kyunki sklearn model ko 2D array (1 row, 25 columns) chahiye hota hai
        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, 
                                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, 
                                f21, f22, f23, f24, f25]])
        
        # Seedha loaded model se prediction maang rahe hain
        prediction = model.predict(input_data)[0]
        
        # Result ko ekdum badhiya style mein print karenge
        st.success(f"### Predicted Fatigue Strength: {prediction:.2f} MPa")
        st.balloons()
# --- TAB 2: OPTIMIZER ---
# --- TAB 2: OPTIMIZER ---
with tab2:
    st.header("Find Best Recipe for Target Strength (Inverse Design)")
    st.markdown("Specify your required strength. The AI will reverse-engineer the exact 25-parameter composition!")
    
    # User target input
    target_mpa = st.number_input("Target Fatigue Strength (MPa)", min_value=200.0, max_value=1200.0, value=650.0, step=10.0)
    
    st.markdown("---")
    st.subheader("🔒 Constraints & Fixed Parameters")
    st.info("Check the box to lock a parameter and set its fixed value. AI will optimize the rest.")
    
    # 1. Master List of all 25 features with their (Name, Min, Max, Default)
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

    fixed_values_dict = {}  # Dictionary to remember which ones user locked
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
                st.markdown("---") # Visual separator
                
    st.write("") # Add some spacing
    
    if st.button("Run AI Optimizer 🚀", use_container_width=True):
        
        # UI Spinner while calculating
        with st.spinner("AI is evaluating millions of combinations. Please wait ~10 seconds..."):
            
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
                maxiter=100,   # Increased from 30 to 100
                popsize=20,    # Increased from 15 to 20
                tol=0.01       # Stricter tolerance
            )
            
            # Extract Results
            best_recipe = result.x
            achieved_strength = model.predict(best_recipe.reshape(1, -1))[0]
            
            st.success(f"### 🎉 Optimization Complete! Achieved Strength: {achieved_strength:.2f} MPa")
            
            st.markdown("#### Complete Optimized Recipe (All 25 Parameters):")
            
            # 5. Display ALL results in a clean 4-column layout
            res_cols = st.columns(4)
            for i, (name, _, _, _) in enumerate(features):
                res_col = res_cols[i % 4]
                with res_col:
                    # Agar user ne lock kiya tha, toh 🔒 icon dikhayenge, warna 🟢
                    if i in fixed_values_dict:
                        st.info(f"🔒 **{name}:**\n{best_recipe[i]:.3f}")
                    else:
                        st.success(f"🟢 **{name}:**\n{best_recipe[i]:.3f}")