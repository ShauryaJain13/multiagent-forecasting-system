from agent.base_agent import BaseAgent


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
        result = super().run(task=task, state=state)
        # state.forecast_metrics = result
        state.mark_agent_complete(self.name)
        return result

    def select_model(self, state):
        """
        Selecting the best model based on the characteristics of the dataset
        and requirements of the forecasting request
        """
        data = state.data
        data_summary = state.data_summary
        models = self._get_potential_models(data_summary)
        results = {}

        for model in models:
            score = self._evaluate_model(model, data, data_summary)
            results[model] = score
        best_model = min(results, key=results.get)

        return best_model, results

    def forecast(self, model, data, horizon):
        """
        This method does the actual forecasting with the given model for the
        given forecast
        """

    def _get_potential_models(self, summary):
        """
        The different basic kinds of forecasting models available
        """
        models = ["naive", "holt-winters", "xgboost"]
        if summary.get("has_seasonality"):
            models.append("sarima")
        return models
