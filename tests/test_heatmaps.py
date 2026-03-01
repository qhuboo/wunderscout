import pytest
import wunderscout


class TestHeatmapGenerator:
    @pytest.fixture(autouse=True)
    def setup(self, tracking_result):
        self.generator = wunderscout.HeatmapGenerator()
        self.tracking_result = tracking_result

    def test_all_players(self):
        players = self.generator.all_players(self.tracking_result)

        assert players

