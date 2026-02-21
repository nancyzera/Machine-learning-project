f
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class brain:
    def __init__(self):
        self.SLOPE_BETA = -0.45
        self.INFRA_BETA = 0.30
        self.INCOME_BETA = 0.15
        self.SAFETY_BETA = 0.20
        self.RAIN_THETA = -0.12
        self.WIND_THETA = -0.08
        self.POLLUTION_THETA = -0.10

class Filter:
    def __init__(self, co, pm10, o3, weights=None):
        self.weights = weights if weights else [0.4, 0.4, 0.2]
        self.co = co
        self.pm10 = pm10
        self.o3 = o3

    def pollution_index(self):
        co_norm = self.co / 100
        pm10_norm = self.pm10 / 200
        o3_norm = self.o3 / 300
        return self.weights[0]*co_norm + self.weights[1]*pm10_norm + self.weights[2]*o3_norm

class Scenario:
    def __init__(self, baseline_trips, brain):
        self.Dsbase = baseline_trips
        self.brain = brain

    def adjusted_demand(self, station, slope=0, infra_score=0, income_level=0,
                        safety_score=0, rain=0, wind=0, PI=0):
        slope_effect = self.brain.SLOPE_BETA * slope
        infra_effect = self.brain.INFRA_BETA * infra_score
        income_effect = self.brain.INCOME_BETA * income_level
        safety_effect = self.brain.SAFETY_BETA * safety_score
        env_effect = self.brain.RAIN_THETA*rain + self.brain.WIND_THETA*wind + self.brain.POLLUTION_THETA*PI
        total_effect = slope_effect + infra_effect + income_effect + safety_effect + env_effect

        baseline = self.Dsbase.get(station, 0)
        Ds_t = baseline * np.exp(total_effect)
        pct_change = ((Ds_t - baseline) / baseline * 100) if baseline != 0 else 0
        return Ds_t, pct_change

    def run_scenario(self, **kwargs):
        results = {}
        for station in self.Dsbase.keys():
            Ds_t, pct = self.adjusted_demand(station, **kwargs)
            results[station] = {"Adjusted Trips": Ds_t, "Change (%)": pct}
        return results

def main():
    st.markdown('<h1 style="color:#FF6F61;"> Kigali Cycling Demand Simulator</h1>', unsafe_allow_html=True)

 
    st.markdown("###  Input Baseline Trips (Dsbase)")
    baseline_trips = {}
    stations = ["Gikondo", "CBD", "Nyabugogo"]
    for s in stations:
        baseline_trips[s] = st.number_input(f"Baseline trips at {s}", value=500, step=50)

   
    st.markdown("###  Input Scenario Parameters")
    slope = st.slider("Road Slope (%)", 0, 20, 5)
    infra_score = st.slider("Infrastructure Accessibility Score", 0, 5, 1)
    income_level = st.slider("Income Level", 0, 5, 1)
    safety_score = st.slider("Safety Score", 0, 5, 1)
    rain = st.number_input("Rainfall (mm)", 0, 100, 10)
    wind = st.number_input("Wind Speed (m/s)", 0, 20, 3)

 
    st.markdown("###  Pollution Data")
    co = st.number_input("CO (ppm)", 0, 100, 30)
    pm10 = st.number_input("PM10 (µg/m³)", 0, 200, 50)
    o3 = st.number_input("O3 (ppb)", 0, 300, 100)

  
    st.markdown("### Model Formula")
    st.latex(r'Ds_t = Ds_{base} \times e^{\beta_{slope} Slope + \beta_{infra} Infra + \beta_{income} Income + \beta_{safety} Safety + \theta_{rain} Rain + \theta_{wind} Wind + \theta_{PI} PI}')
    st.markdown("""
- **Slope**: Road gradient (%)
- **Infra**: Infrastructure score
- **Income**: Socio-economic factor
- **Safety**: Safety perception score
- **Rain**: Rainfall in mm
- **Wind**: Wind speed
- **PI**: Pollution Index computed from CO, PM10, O3
""")

    if st.button("Run Scenario"):
        brain = Brain()
        filter_env = Filter(co, pm10, o3)
        PI = filter_env.pollution_index()
        scenario = Scenario(baseline_trips, brain)
        results = scenario.run_scenario(
            slope=slope, infra_score=infra_score, income_level=income_level,
            safety_score=safety_score, rain=rain, wind=wind, PI=PI
        )

     
        st.markdown('<h3 style="color:#4CAF50;">### Simulation Results</h3>', unsafe_allow_html=True)
        for station, data in results.items():
            st.write(f"**{station}**: {data['Adjusted Trips']:.1f} trips ({data['Change (%)']:.2f}% change)")

    
        stations_list = list(results.keys())
        trips = [data['Adjusted Trips'] for data in results.values()]

        fig, ax = plt.subplots()
        ax.bar(stations_list, trips, color=['#4CAF50', '#2196F3', '#FF9800'])
        ax.set_ylabel("Adjusted Trips")
        ax.set_title("Adjusted Cycling Trips per Station")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(stations_list, trips, marker='o', linestyle='--', color='purple')
        ax2.set_ylabel("Adjusted Trips")
        ax2.set_title("Cycling Demand Trend")
        st.pyplot(fig2)

        
        st.markdown("###  AI Assistant Mode (Coming Soon)")
        question = st.text_input("Ask about your cycling demand scenario:")
        if question:
            st.write("AI Mode will answer your questions here (under development).")
           
def calculate_demand(baseline_trips, slope=0, infra_score=0, income_level=0,
                     safety_score=0, rain=0, wind=0, co=0, pm10=0, o3=0):
    brain = Brain()
    filter_env = Filter(co, pm10, o3)
    PI = filter_env.pollution_index()
    scenario = Scenario(baseline_trips, brain)
    results = scenario.run_scenario(
        slope=slope,
        infra_score=infra_score,
        income_level=income_level,
        safety_score=safety_score,
        rain=rain,
        wind=wind,
        PI=PI
    )
    return results


if __name__ == "__main__":
    main()
