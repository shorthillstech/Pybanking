import subprocess

class Analysis:
    def sweetviz_analysis(self, df):
        subprocess.run(['pip', 'install', 'sweetviz'])
        import sweetviz as sv
        return sv.analyze(df).show_html()

    def dataprep_analysis(self, df):
        subprocess.run(['pip', 'install', 'dataprep'])
        from dataprep.eda import create_report
        report = create_report(df)
        return report.save('DataPrep_Report.html')
    
    def pandas_analysis(self, df):
        subprocess.run(['pip', 'install', 'pandas-profiling'])
        from pandas_profiling import ProfileReport
        return ProfileReport(df).to_file("Pandas_Profiling_Report.html")
