class ModelAgreement:

    def calculate(self, plugin_outputs):

        if not plugin_outputs:
            return 0.0

        def avg(values):
            return sum(values) / len(values) if values else 0.0

        home_values = [float(x.get("home_win", 0.0)) for x in plugin_outputs]
        draw_values = [float(x.get("draw", 0.0)) for x in plugin_outputs]
        away_values = [float(x.get("away_win", 0.0)) for x in plugin_outputs]

        def dispersion(values):
            if not values:
                return 1.0
            mean_value = avg(values)
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            return variance**0.5

        home_disp = dispersion(home_values)
        draw_disp = dispersion(draw_values)
        away_disp = dispersion(away_values)

        mean_disp = (home_disp + draw_disp + away_disp) / 3

        agreement = 1.0 - mean_disp * 3
        agreement = max(0.0, min(1.0, agreement))

        return agreement
