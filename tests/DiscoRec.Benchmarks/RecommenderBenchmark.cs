using BenchmarkDotNet.Attributes;

namespace DiscoRec.Benchmarks;

[MemoryDiagnoser(false)]
public class RecommenderBenchmark
{
    private Dataset<int, string>? data;

    [GlobalSetup]
    public async Task GlobalSetup()
    {
        data = await Data.LoadMovieLens();
    }

    [Benchmark]
    public void Recommender()
    {
        var recommender = Recommender<int, string>.FitExplicit(data!, new RecommenderOptions { Factors = 20 });
        foreach (var id in recommender.UserIds())
        {
            recommender.UserRecs(id, 5);
        }
        foreach (var id in recommender.ItemIds())
        {
            recommender.ItemRecs(id, 5);
        }
    }
}
