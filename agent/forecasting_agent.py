from agent.base_agent import BaseAgent
from forecasting.evaluation import evaluate_model
# from forecasting.models import NaiveModel, HoltWinters, XGBoostRegressor


class ForecastingAgent(BaseAgent):
    """
    This is an agent that enables the actual forecasting of the data
    """

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(name="Forecasting Agent", llm=llm, tools=tools,
                         system_prompt="You are a forecasting agent. You must"
                         "select the correct forecasting model, prepare the"
                         "data appropriately, interpret the forecast"
                         "correctly, and conduct all necessary tasks necessary"
                         "to forecast accurately",
                         prompt_builder=prompt_builder, memory=memory)

    def run(self, task, state):
        """
        Processes and runs the forecasting request
        """
        # result = super().run(task=task, state=state)
        best_model, result = self.select_model(state)
        state.forecast_metrics = result
        prediction = self.forecast(best_model, state.data, self.horizon)
        state.forecast = prediction
        # state.forecast_metrics = result
        state.mark_agent_complete(self.name)
        return prediction  # result

    def select_model(self, state):
        """
        Selecting the best model based on the characteristics of the dataset
        and requirements of the forecasting request
        """
        data = state.data
        # data_summary = state.data_summary
        models = self._get_potential_models(state.data_summary)

        split_index = int(0.8 * len(data))
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        best_model, results = evaluate_model(models, train_data, test_data)
        return best_model, results

        # results = {}

        # for model in models:
        #     score = evaluate_model(model, data, state.data_summary)
        #     results[model] = score
        # best_model = min(results, key=results.get)

        # return best_model, results

    def forecast(self, model, data, horizon):
        """
        This method does the actual forecasting with the given model for the
        given forecast
        """
        model.fit(data)
        prediction = model.forecast(horizon)
        return prediction
        # forecast = model.forecast(data, horizon)
        # return forecast

    def _get_potential_models(self, summary):
        """
        The different basic kinds of forecasting models available
        """
        models = ["naive", "holt-winters", "xgboost", "sarima"]
        # if summary.get("has_seasonality"):
        # models.append("sarima")
        return models
