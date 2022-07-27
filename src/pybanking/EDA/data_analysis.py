from fileinput import filename
import subprocess

class Analysis:
    def sweetviz_analysis(self, df):
        subprocess.run(['pip', 'install', 'sweetviz'])
        import sweetviz as sv
        return sv.analyze(df).show_html()

    def dataprep_analysis(self, df):
        subprocess.run(['pip', 'install', 'dataprep'])
        from dataprep.eda import plot, create_report
        return create_report(df).save(filename='Dataprep_report.html')
    
    def pandas_analysis(self, df):
        subprocess.run(['pip', 'install', 'pandas-profiling'])
        from pandas_profiling import ProfileReport
        return ProfileReport(df).to_file("Profiling_report.html")
