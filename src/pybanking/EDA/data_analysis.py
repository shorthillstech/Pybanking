import subprocess

from pydantic import FilePath

class Analysis:
    def sweetviz_analysis(self, df):
        subprocess.run(['pip', 'install', 'sweetviz'])
        import sweetviz as sv
        if df.shape[1]>50:
            return sv.analyze(df, pairwise_analysis='off')
        return sv.analyze(df) 

    def dataprep_analysis(self, df):
        subprocess.run(['pip', 'install', 'dataprep'])
        from dataprep.eda import create_report
        if df.shape[1]>50:
            return create_report(df, mode="basic")
        return create_report(df)
    
    def pandas_analysis(self, df):
        subprocess.run(['pip', 'install', 'pandas-profiling'])
        from pandas_profiling import ProfileReport
        if df.shape[1]>50:
            return ProfileReport(df, minimal=True)
        return ProfileReport(df)
