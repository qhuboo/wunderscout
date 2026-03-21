import json
import pytest
import wunderscout


class TestHeatmapGenerator:
    """Test core HeatmapGenerator functionality."""

    @pytest.fixture
    def generator(self):
        return wunderscout.HeatmapGenerator()

    @pytest.fixture
    def custom_generator(self):
        return wunderscout.HeatmapGenerator(
            pitch_length=120.0,
            pitch_width=80.0,
            histogram_bins=(25, 25),
            kde_grid_size=(50, 50),
        )


class TestPlayerHeatmaps(TestHeatmapGenerator):
    """Test player-specific heatmap generation."""

    @pytest.mark.parametrize(
        "heatmap_type,should_have,should_not_have",
        [
            ("histogram", ["histogram"], ["kde"]),
            ("kde", ["kde"], ["histogram"]),
            ("both", ["histogram", "kde"], []),
        ],
        ids=["histogram-only", "kde-only", "both-types"],
    )
    def test_player_heatmaps(
        self, generator, frames, heatmap_type, should_have, should_not_have
    ):
        """Test player heatmap with each heatmap type options."""
        player_ids = list(frames.get_all_player_ids())
        result = generator.player(frames, player_ids[0], heatmap_type=heatmap_type)

        assert isinstance(result, wunderscout.Heatmap)
        assert result.identifier == player_ids[0]
        assert result.prefix == "player"

        for key in should_have:
            assert key in result.data
        for key in should_not_have:
            assert key not in result.data

    def test_histogram_structure(self, generator, frames):
        """Test histogram data structure is correct."""
        player_ids = list(frames.get_all_player_ids())
        result = generator.player(frames, player_ids[0], heatmap_type="histogram")

        hist = result.data["histogram"]
        assert "xedges" in hist
        assert "yedges" in hist
        assert "values" in hist
        assert len(hist["xedges"]) == generator.histogram_bins[0] + 1
        assert len(hist["yedges"]) == generator.histogram_bins[1] + 1
        assert len(hist["values"]) == generator.histogram_bins[1]
        assert len(hist["values"][0]) == generator.histogram_bins[0]

    def test_kde_structure(self, generator, frames):
        """Test KDE data structure is correct."""
        player_ids = list(frames.get_all_player_ids())
        result = generator.player(frames, player_ids[0], heatmap_type="kde")

        kde = result.data["kde"]
        assert "x" in kde
        assert "y" in kde
        assert "values" in kde
        assert len(kde["x"]) == generator.kde_grid_size[0]
        assert len(kde["y"]) == generator.kde_grid_size[1]
        assert len(kde["values"]) == generator.kde_grid_size[1]
        assert len(kde["values"][0]) == generator.kde_grid_size[0]

    def test_custom_generator_dimensions(self, custom_generator, frames):
        """Test that custom generator parameters are respected."""
        player_ids = list(frames.get_all_player_ids())
        result = custom_generator.player(frames, player_ids[0])

        hist = result.data["histogram"]
        assert len(hist["xedges"]) == 26  # bins + 1
        assert len(hist["yedges"]) == 26

        kde = result.data["kde"]
        assert len(kde["x"]) == 50
        assert len(kde["y"]) == 50


class TestTeamHeatmaps(TestHeatmapGenerator):
    """Test team-specific heatmap generation."""

    @pytest.mark.parametrize(
        "heatmap_type, should_have, should_not_have",
        [
            ("histogram", ["histogram"], ["kde"]),
            ("kde", ["kde"], ["histogram"]),
            ("both", ["histogram", "kde"], []),
        ],
        ids=["histogram-only", "kde-only", "both-types"],
    )
    def test_team_heatmaps(
        self, generator, frames, heatmap_type, should_have, should_not_have
    ):
        """Test team heatmap with all heatmap type options."""
        team_ids = list(frames.get_all_team_ids())
        result = generator.team(frames, team_ids[0], heatmap_type=heatmap_type)

        assert isinstance(result, wunderscout.Heatmap)
        assert result.identifier == team_ids[0]
        assert result.prefix == "team"

        for key in should_have:
            assert key in result.data
        for key in should_not_have:
            assert key not in result.data

    def test_different_teams_different_results(self, generator, frames):
        """Test that different teams generate different heatmaps."""
        team_ids = list(frames.get_all_team_ids())
        if len(team_ids) == 2:
            team_0 = generator.team(frames, team_ids[0])
            team_1 = generator.team(frames, team_ids[1])

            assert team_0.identifier != team_1.identifier
            assert team_0.data != team_1.data  # Does this actually deep check values


class TestHeatmapSave:
    """Test heatmap saving functionality."""

    @pytest.fixture
    def sample_player_heatmap_both(self):
        """Player heatmap with both histogram and kde."""

        return wunderscout.Heatmap(
            data={
                "histogram": {"xedges": [0, 1], "yedges": [0, 1], "values": [[5]]},
                "kde": {"x": [0, 1], "y": [0, 1], "values": [[0.1, 0.2]]},
            },
            identifier=42,
            prefix="player",
        )

    @pytest.fixture
    def sample_player_heatmap_histogram_only(self):
        """Player heatmap with histogram only."""

        return wunderscout.Heatmap(
            data={
                "histogram": {"xedges": [0, 1], "yedges": [0, 1], "values": [[5]]},
            },
            identifier=24,
            prefix="player",
        )

    @pytest.mark.parametrize(
        "heatmap_type, files_created",
        [
            ("histogram", ["player_42_histogram.json"]),
            ("kde", ["player_42_kde.json"]),
            ("both", ["player_42_histogram.json", "player_42_kde.json"]),
        ],
        ids=["histogram-only", "kde-only", "both-types"],
    )
    def test_save(
        self, tmp_path, sample_player_heatmap_both, heatmap_type, files_created
    ):
        """Test saving with all heatmap type options."""

        result = sample_player_heatmap_both.save(tmp_path, heatmap_type=heatmap_type)

        assert len(result.successful_paths) == len(files_created)
        assert len(result.failed_paths) == 0
        assert len(result.errors) == 0

        for filename in files_created:
            assert (tmp_path / filename).exists()
            assert (tmp_path / filename) in result.successful_paths

    def test_save_with_missing_data(
        self, tmp_path, sample_player_heatmap_histogram_only
    ):
        """Test saving when requested type doesn't exist in data."""

        result = sample_player_heatmap_histogram_only.save(tmp_path, heatmap_type="kde")

        assert len(result.successful_paths) == 0
        assert len(result.failed_paths) == 1
        assert len(result.errors) == 1

    def test_save_creates_directory(self, tmp_path, sample_player_heatmap_both):
        """Test save creates nested directories if they don't exist."""

        nested_path = tmp_path / "level1" / "level2" / "level3"
        result = sample_player_heatmap_both.save(nested_path)

        assert nested_path.exists()
        assert len(result.successful_paths) == 2

    def test_save_handles_write_error(self, tmp_path, sample_player_heatmap_both):
        """Test save handles file write errors gracefully."""

        # Create a read-only directory to force a write error
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            result = sample_player_heatmap_both.save(readonly_dir, heatmap_type="both")

            assert len(result.failed_paths) >= 1
            assert len(result.errors) >= 1
        finally:
            # Cleanup: restore permissions
            readonly_dir.chmod(0o755)

    def test_save_type_error(self, tmp_path):
        """Test saving where data object contains non JSON serializable data."""

        heatmap = wunderscout.Heatmap(
            data={
                "histogram": {"values": {1, 2, 3}}
            },  # Values contains a set {1, 2, 3} which is not JSON serializable
            identifier=23,
            prefix="player",
        )

        result = heatmap.save(tmp_path, heatmap_type="histogram")

        assert len(result.failed_paths) == 1
        assert len(result.errors) == 1


@pytest.mark.integration
class TestIntegration(TestHeatmapGenerator):
    """Integration tests with real Frames objects."""

    def test_end_to_end_player(self, generator, frames, tmp_path):
        player_ids = list(frames.get_all_player_ids())

        if not player_ids:
            pytest.skip("No players in test frames")

        player_id = player_ids[0]
        heatmap = generator.player(frames, player_id, heatmap_type="both")
        result = heatmap.save(tmp_path, heatmap_type="both")

        assert len(result.successful_paths) > 0
        assert len(result.failed_paths) == 0
        assert len(result.errors) == 0

        for path in result.successful_paths:
            assert path.exists()
            with open(path, "r") as f:
                data = json.load(f)
                assert isinstance(data, dict)

    def test_end_to_end_team(self, generator, frames, tmp_path):
        team_ids = list(frames.get_all_team_ids())

        if not team_ids:
            pytest.skip("No teams in test frames")

        team_id = team_ids[0]
        heatmap = generator.team(frames, team_id, heatmap_type="both")
        result = heatmap.save(tmp_path, heatmap_type="both")

        assert len(result.successful_paths) > 0
        assert len(result.failed_paths) == 0
        assert len(result.errors) == 0

        for path in result.successful_paths:
            assert path.exists()
            with open(path, "r") as f:
                data = json.load(f)
                assert isinstance(data, dict)
