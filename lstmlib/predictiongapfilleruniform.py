from lstmlib.predictiongapfillerbase import PredictionGapFillerBase

class PredictionGapFillerUniform(PredictionGapFillerBase):
    def get(self, time_stamp: str, features: list) -> float:
        return features[0]