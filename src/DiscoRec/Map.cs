using System;
using System.Collections.Generic;

namespace DiscoRec;

internal class Map<T> where T : notnull
{
    private Dictionary<T, int> Mapping;
    private List<T> Vec;

    public Map()
    {
        Mapping = new Dictionary<T, int>();
        Vec = new List<T>();
    }

    public int Count
    {
        get => Mapping.Count;
    }

    public int Add(T id)
    {
        int i;
        if (!Mapping.TryGetValue(id, out i))
        {
            i = Mapping.Count;
            Mapping.Add(id, i);
            Vec.Add(id);
        }
        return i;
    }

    public int? Get(T id)
    {
        int i;
        if (Mapping.TryGetValue(id, out i))
            return i;
        return null;
    }

    public T Lookup(int index)
        => Vec[index];

    public T[] Ids()
        => Vec.ToArray();
}
