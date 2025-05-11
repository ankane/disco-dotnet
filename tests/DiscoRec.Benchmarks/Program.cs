using BenchmarkDotNet.Running;

namespace DiscoRec.Benchmarks;

public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run<RecommenderBenchmark>();
    }
}
