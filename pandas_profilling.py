from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("c:/Users/maury/Downloads/InsightAI_Output/final_output.csv")

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("report.html")