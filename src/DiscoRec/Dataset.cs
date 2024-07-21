namespace DiscoRec;

using System;
using System.Collections.Generic;
using System.Linq;

internal class Rating<T, U> where T : notnull where U : notnull
{
    public T UserId;
    public U ItemId;
    public float Value;

    public Rating(T userId, U itemId, float value)
    {
        UserId = userId;
        ItemId = itemId;
        Value = value;
    }
}

public class Dataset<T, U> where T : notnull where U : notnull
{
    internal List<Rating<T, U>> Data;

    public Dataset()
        => Data = new List<Rating<T, U>>();

    public Dataset(int capacity)
        => Data = new List<Rating<T, U>>(capacity);

    public int Count
    {
        get => Data.Count;
    }

    public void Add(T userId, U itemId, float value)
        => Data.Add(new Rating<T, U>(userId, itemId, value));

    public void Add(T userId, U itemId)
        => Add(userId, itemId, 1);

    public (Dataset<T, U>, Dataset<T, U>) SplitRandom(float p)
    {
        var index = (int)(p * Data.Count);
        var rand = new Random();
        // TODO improve
        var data = Data.OrderBy((v) => rand.Next()).ToList();
        var trainSet = data.Take(index).ToList();
        var validSet = data.Skip(index).ToList();
        return (new Dataset<T, U> { Data = trainSet }, new Dataset<T, U> { Data = validSet });
    }
}
