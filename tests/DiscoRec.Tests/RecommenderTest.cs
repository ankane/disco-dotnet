namespace DiscoRec.Tests;

public class RecommenderTest
{
    [Fact]
    public async Task TestExplicit()
    {
        var data = await Data.LoadMovieLens();
        var recommender = Recommender<int, string>.FitExplicit(data, new RecommenderOptions { Factors = 20 });

        Assert.Equal(3.52986, recommender.GlobalMean(), 0.00001);

        var recs = recommender.ItemRecs("Star Wars (1977)", 5);
        Assert.Equal(5, recs.Length);

        var itemIds = recs.Select((r) => r.Id).ToList();
        Assert.Contains("Empire Strikes Back, The (1980)", itemIds);
        Assert.Contains("Return of the Jedi (1983)", itemIds);
        Assert.DoesNotContain("Star Wars (1977)", itemIds);

        Assert.Equal(0.9972, recs[0].Score, 0.01);
    }

    [Fact]
    public async Task TestImplicit()
    {
        var data = await Data.LoadMovieLens();
        var recommender = Recommender<int, string>.FitImplicit(data, new RecommenderOptions { Factors = 20 });

        Assert.Equal(0, recommender.GlobalMean());

        var recs = recommender.ItemRecs("Star Wars (1977)", 5);
        var itemIds = recs.Select((r) => r.Id).ToList();
        Assert.Contains("Return of the Jedi (1983)", itemIds);
        Assert.DoesNotContain("Star Wars (1977)", itemIds);
    }

    [Fact]
    public void TestRated()
    {
        var data = new Dataset<int, string>();
        data.Add(1, "A", 1.0f);
        data.Add(1, "B", 1.0f);
        data.Add(1, "C", 1.0f);
        data.Add(1, "D", 1.0f);
        data.Add(2, "C", 1.0f);
        data.Add(2, "D", 1.0f);
        data.Add(2, "E", 1.0f);
        data.Add(2, "F", 1.0f);

        var recommender = Recommender<int, string>.FitExplicit(data);

        var itemIds = recommender.UserRecs(1, 5).Select((r) => r.Id).OrderBy((v) => v).ToList();
        Assert.Equal(new string[] { "E", "F" }, itemIds);

        itemIds = recommender.UserRecs(2, 5).Select((r) => r.Id).OrderBy((v) => v).ToList();
        Assert.Equal(new string[] { "A", "B" }, itemIds);
    }

    [Fact]
    public void TestItemRecsSameScore()
    {
        var data = new Dataset<int, string>();
        data.Add(1, "A");
        data.Add(1, "B");
        data.Add(2, "C");

        var recommender = Recommender<int, string>.FitImplicit(data);

        var itemIds = recommender.ItemRecs("A", 5).Select((r) => r.Id).ToList();
        Assert.Equal(new string[] { "B", "C" }, itemIds);
    }

    [Fact]
    public async Task TestSimilarUsers()
    {
        var data = await Data.LoadMovieLens();
        var recommender = Recommender<int, string>.FitExplicit(data);

        Assert.Equal(5, recommender.SimilarUsers(1, 5).Length);
        Assert.Empty(recommender.SimilarUsers(10000, 5));
    }

    [Fact]
    public void TestIds()
    {
        var data = new Dataset<int, string>();
        data.Add(1, "A", 1.0f);
        data.Add(1, "B", 1.0f);
        data.Add(2, "B", 1.0f);

        var recommender = Recommender<int, string>.FitExplicit(data);
        Assert.Equal(new int[] { 1, 2 }, recommender.UserIds());
        Assert.Equal(new string[] { "A", "B" }, recommender.ItemIds());
    }

    [Fact]
    public void TestFactors()
    {
        var data = new Dataset<int, string>();
        data.Add(1, "A", 1.0f);
        data.Add(1, "B", 1.0f);
        data.Add(2, "B", 1.0f);

        var recommender = Recommender<int, string>.FitExplicit(data, new RecommenderOptions { Factors = 20 });

        Assert.Equal(20, recommender.UserFactors(1)!.Length);
        Assert.Equal(20, recommender.ItemFactors("A")!.Length);

        Assert.Null(recommender.UserFactors(3));
        Assert.Null(recommender.ItemFactors("C"));
    }

    [Fact]
    public async Task TestValidationSetExplicit()
    {
        var data = await Data.LoadMovieLens();

        var (trainSet, validSet) = data.SplitRandom(0.8f);
        Assert.Equal(80000, trainSet.Count);
        Assert.Equal(20000, validSet.Count);
        Recommender<int, string>.FitExplicit(trainSet, validSet, new RecommenderOptions { Factors = 20, Verbose = false });
    }

    [Fact]
    public void TestUserRecsNewUser()
    {
        var data = new Dataset<int, int>();
        data.Add(1, 1, 5.0f);
        data.Add(2, 1, 3.0f);

        var recommender = Recommender<int, int>.FitExplicit(data);
        Assert.Empty(recommender.UserRecs(1000, 5));
    }

    [Fact]
    public async Task TestPredict()
    {
        var data = await Data.LoadMovieLens();
        var recommender = Recommender<int, string>.FitExplicit(data);
        recommender.Predict(1, "Star Wars (1977)");
    }

    [Fact]
    public async Task TestPredictNewUser()
    {
        var data = await Data.LoadMovieLens();
        var recommender = Recommender<int, string>.FitExplicit(data, new RecommenderOptions { Factors = 20 });
        Assert.Equal(recommender.GlobalMean(), recommender.Predict(100000, "Star Wars (1977)"), 0.001);
    }

    [Fact]
    public async Task TestPredictNewItem()
    {
        var data = await Data.LoadMovieLens();

        var recommender = Recommender<int, string>.FitExplicit(data, new RecommenderOptions { Factors = 20 });
        Assert.Equal(recommender.GlobalMean(), recommender.Predict(1, "New movie"), 0.001);
    }

    [Fact]
    public void TestNoTrainingData()
    {
        var data = new Dataset<int, string>();
        var error = Assert.Throws<ArgumentException>(() => Recommender<int, string>.FitExplicit(data));
        Assert.Equal("No training data", error.Message);
    }
}
