using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DiscoRec;

public class RecommenderOptions
{
    public int Factors = 8;
    public int Iterations = 20;
    public bool? Verbose = null;
}

public class Rec<T> where T : notnull
{
    public T Id;
    public float Score;

    internal Rec(T id, float score)
    {
        Id = id;
        Score = score;
    }
}

internal class MatrixValue
{
    public uint RowIndex;
    public uint ColumnIndex;
    public float Label;

    public MatrixValue(int rowIndex, int columnIndex, float label)
    {
        RowIndex = (uint)rowIndex;
        ColumnIndex = (uint)columnIndex;
        Label = label;
    }
}

public class Recommender<T, U> where T : notnull where U : notnull
{
    private Map<T> UserMap;
    private Map<U> ItemMap;
    private Dictionary<int, HashSet<int>> Rated;
    private float globalMean;
    private Matrix userFactors;
    private Matrix itemFactors;
    private float[]? UserNorms;
    private float[]? ItemNorms;

    private Recommender(Map<T> userMap, Map<U> itemMap, Dictionary<int, HashSet<int>> rated, float globalMean, Matrix userFactors, Matrix itemFactors)
    {
        UserMap = userMap;
        ItemMap = itemMap;
        Rated = rated;
        this.globalMean = globalMean;
        this.userFactors = userFactors;
        this.itemFactors = itemFactors;
    }

    public static Recommender<T, U> FitExplicit(Dataset<T, U> trainSet)
        => Fit(trainSet, null, new RecommenderOptions(), false);

    public static Recommender<T, U> FitExplicit(Dataset<T, U> trainSet, RecommenderOptions options)
        => Fit(trainSet, null, options, false);

    public static Recommender<T, U> FitExplicit(Dataset<T, U> trainSet, Dataset<T, U> validSet)
        => Fit(trainSet, validSet, new RecommenderOptions(), false);

    public static Recommender<T, U> FitExplicit(Dataset<T, U> trainSet, Dataset<T, U> validSet, RecommenderOptions options)
        => Fit(trainSet, validSet, options, false);

    public static Recommender<T, U> FitImplicit(Dataset<T, U> trainSet)
        => Fit(trainSet, null, new RecommenderOptions(), true);

    public static Recommender<T, U> FitImplicit(Dataset<T, U> trainSet, RecommenderOptions options)
        => Fit(trainSet, null, options, true);

    public static Recommender<T, U> FitImplicit(Dataset<T, U> trainSet, Dataset<T, U> validSet)
        => Fit(trainSet, validSet, new RecommenderOptions(), true);

    public static Recommender<T, U> FitImplicit(Dataset<T, U> trainSet, Dataset<T, U> validSet, RecommenderOptions options)
        => Fit(trainSet, validSet, options, true);

    private static Recommender<T, U> Fit(Dataset<T, U> trainSet, Dataset<T, U>? validSet, RecommenderOptions options, bool isImplicit)
    {
        if (!trainSet.Data.Any())
            throw new ArgumentException("No training data");

        var userMap = new Map<T>();
        var itemMap = new Map<U>();
        var rated = new Dictionary<int, HashSet<int>>();

        var input = new List<MatrixValue>(trainSet.Count);
        foreach (var v in trainSet.Data)
        {
            var u = userMap.Add(v.UserId);
            // TODO improve
            if (u == userMap.Count - 1)
                rated.TryAdd(u, new HashSet<int>());
            var i = itemMap.Add(v.ItemId);
            input.Add(new MatrixValue(u, i, v.Value));
            rated[u].Add(i);
        }

        var users = userMap.Count;
        var items = itemMap.Count;

        var globalMean = isImplicit ? 0 : input.Sum((v) => v.Label) / input.Count;

        MLContext mlContext = new MLContext();
        // for debugging
        // mlContext.Log += (sender, e) => Console.WriteLine(e.Message);
        var schema = SchemaDefinition.Create(typeof(MatrixValue));
        schema[0].ColumnType = new KeyDataViewType(typeof(uint), users);
        schema[1].ColumnType = new KeyDataViewType(typeof(uint), items);
        IDataView trainView = mlContext.Data.LoadFromEnumerable(input, schema);
        IDataView? validView = null;
        if (validSet != null)
        {
            var validInput = new List<MatrixValue>(validSet.Count);
            foreach (var v in validSet.Data)
            {
                var u = userMap.Get(v.UserId) ?? int.MaxValue;
                var i = itemMap.Get(v.ItemId) ?? int.MaxValue;
                validInput.Add(new MatrixValue(u, i, v.Value));
            }
            validView = mlContext.Data.LoadFromEnumerable(validInput, schema);
        }

        var factors = options.Factors;
        var loss = isImplicit ? MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass : MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression;
        var verbose = options.Verbose ?? validSet != null;

        var trainerOptions = new MatrixFactorizationTrainer.Options
        {
            MatrixRowIndexColumnName = schema[0].ColumnName,
            MatrixColumnIndexColumnName = schema[1].ColumnName,
            LabelColumnName = schema[2].ColumnName,
            NumberOfIterations = options.Iterations,
            ApproximationRank = factors,
            Quiet = !verbose,
            LossFunction = loss
        };
        var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(trainerOptions);
        var model = trainer.Fit(trainView, validView);

        var userFactors = new Matrix(users, factors, model.Model.LeftFactorMatrix.ToArray());
        var itemFactors = new Matrix(items, factors, model.Model.RightFactorMatrix.ToArray());

        return new Recommender<T, U>(userMap, itemMap, rated, globalMean, userFactors, itemFactors);
    }

    public Rec<U>[] UserRecs(T userId, int count)
    {
        var u = UserMap.Get(userId);
        if (u == null)
            return new Rec<U>[] { };

        var rated = Rated[u.Value];
        var factors = userFactors.Row((int)u.Value);

        var predictions = new List<Rec<int>>(itemFactors.Rows);
        for (var j = 0; j < itemFactors.Rows; j++)
            predictions.Add(new Rec<int>(j, Dot(factors, itemFactors.Row(j))));
        predictions = predictions.OrderBy((v) => -v.Score).Take(count + rated.Count).ToList();

        var recs = new List<Rec<U>>(count + rated.Count);
        foreach (var prediction in predictions)
        {
            if (!rated.Contains(prediction.Id))
                recs.Add(new Rec<U>(ItemMap.Ids()[prediction.Id], prediction.Score));
        }
        return recs.Take(count).ToArray();
    }

    public Rec<U>[] ItemRecs(U itemId, int count)
    {
        ItemNorms ??= itemFactors.Norms();
        return Similar(ItemMap, itemFactors, ItemNorms, itemId, count);
    }

    public Rec<T>[] SimilarUsers(T userId, int count)
    {
        UserNorms ??= userFactors.Norms();
        return Similar(UserMap, userFactors, UserNorms, userId, count);
    }

    public float Predict(T userId, U itemId)
    {
        var u = UserMap.Get(userId);
        if (u == null)
            return globalMean;

        var i = ItemMap.Get(itemId);
        if (i == null)
            return globalMean;

        return Dot(userFactors.Row((int)u), itemFactors.Row((int)i));
    }

    public T[] UserIds()
        => UserMap.Ids();

    public U[] ItemIds()
        => ItemMap.Ids();

    public float[]? UserFactors(T userId)
    {
        var i = UserMap.Get(userId);
        if (i == null)
            return null;

        return userFactors.Row((int)i).ToArray();
    }

    public float[]? ItemFactors(U itemId)
    {
        var i = ItemMap.Get(itemId);
        if (i == null)
            return null;

        return itemFactors.Row((int)i).ToArray();
    }

    public float GlobalMean()
        => globalMean;

    private Rec<V>[] Similar<V>(Map<V> map, Matrix factors, float[] norms, V id, int count) where V : notnull
    {
        var i = map.Get(id);
        if (i == null)
            return new Rec<V>[] { };

        var rowFactors = factors.Row(i.Value);
        var rowNorm = norms[i.Value];

        var predictions = new List<Rec<int>>(factors.Rows);
        for (var j = 0; j < factors.Rows; j++)
        {
            var denom = rowNorm * norms[j];
            if (denom == 0)
                denom = 0.00001f;
            predictions.Add(new Rec<int>(j, Dot(rowFactors, factors.Row(j)) / denom));
        }
        predictions = predictions.OrderBy((v) => -v.Score).Take(count + 1).ToList();

        var recs = new List<Rec<V>>(count + 1);
        foreach (var prediction in predictions)
        {
            if (prediction.Id != i)
                recs.Add(new Rec<V>(map.Lookup(prediction.Id), prediction.Score));
        }
        return recs.Take(count).ToArray();
    }

    private float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = 0.0f;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}
