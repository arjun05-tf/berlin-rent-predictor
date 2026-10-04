"""Streamlit UI for BerlinRentML. Run: streamlit run app/streamlit_app.py"""

import streamlit as st

from berlinrentml.config import MODELS_DIR
from berlinrentml.inference import predict_rent
from berlinrentml.modeling.training import load_model_artifacts

st.set_page_config(page_title="Berlin Rent Predictor", page_icon="🏠")


@st.cache_resource
def load():
    model, preprocessor, feature_names = load_model_artifacts("final_model", MODELS_DIR)
    cats = {
        f: [str(v) for v in c]
        for f, c in zip(preprocessor.categorical_features, preprocessor.preprocessor.named_transformers_["cat"][-1].categories_)
    }
    return model, preprocessor, feature_names, cats


st.title("🏠 Berlin Rent Predictor")
st.caption("LightGBM trained on ~10k real ImmoScout24 Berlin listings (2018-2020). Historical rents, not today's market.")

try:
    model, preprocessor, feature_names, cats = load()
except FileNotFoundError:
    st.error("No trained model found. Run `python scripts/train.py` first.")
    st.stop()

col1, col2 = st.columns(2)
space = col1.number_input("Living space (m²)", 10.0, 500.0, 60.0, 1.0)
rooms = col2.number_input("Rooms", 0.5, 20.0, 2.0, 0.5)
district = col1.selectbox("District (Ortsteil)", sorted(cats["geo_bln"]), index=None, placeholder="Choose...")
plz = col2.selectbox("Postal code", sorted(cats["geo_plz"]), index=None, placeholder="Choose...")
condition = col1.selectbox("Condition", sorted(cats["condition"]), index=None, placeholder="Unknown")
heating = col2.selectbox("Heating", sorted(cats["heatingType"]), index=None, placeholder="Unknown")
floor = col1.number_input("Floor", 0, 50, 2)
year = col2.number_input("Year constructed", 1800, 2030, 1990)
kitchen, balcony, garden, cellar = (
    c.checkbox(t) for c, t in zip(st.columns(4), ["Kitchen", "Balcony", "Garden", "Cellar"])
)

if st.button("Predict rent", type="primary"):
    rent = predict_rent(
        {
            "livingSpace": space, "rooms": rooms, "floor": floor, "yearConstructed": year,
            "geo_plz": plz, "geo_bln": district, "condition": condition, "heatingType": heating,
            "hasKitchen": kitchen, "hasBalcony": balcony, "hasGarden": garden, "cellar": cellar,
        },
        model, preprocessor, feature_names,
    )
    st.metric("Predicted cold rent (€/month)", f"{rent:,.0f} €")
    st.caption(f"≈ {rent / space:.1f} €/m². Typical error is about ±€190 (grouped-split MAE).")
