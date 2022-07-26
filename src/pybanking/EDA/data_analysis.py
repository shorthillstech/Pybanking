import subprocess

class Analysis:
    def sweetviz_analysis(self, df):
        subprocess.run(['pip', 'install', 'sweetviz'])
        import sweetviz as sv
        return sv.analyze(df).show_html()

    def dataprep_analysis(self, df):
        subprocess.run(['pip', 'install', 'dataprep'])
        from dataprep.eda import plot, create_report
        return create_report(df).show_browser(), plot(df).show_browser()
    
    def pandas_analysis(self, df):
        subprocess.run(['pip', 'install', 'pandas-profiling'])
        from pandas_profiling import ProfileReport
        import webbrowser
        ProfileReport(df).to_file("report.html")
        return webbrowser.open_new_tab("report.html")
