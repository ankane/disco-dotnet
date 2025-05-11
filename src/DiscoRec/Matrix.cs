using System;

namespace DiscoRec;

internal class Matrix
{
    internal int Rows;
    private int Cols;
    private ReadOnlyMemory<float> Data;

    public Matrix(int rows, int cols, ReadOnlyMemory<float> data)
    {
        Rows = rows;
        Cols = cols;
        Data = data;
    }

    public ReadOnlySpan<float> Row(int row)
    {
        var start = row * Cols;
        return Data.Span.Slice(start, Cols);
    }

    public float[] Norms()
    {
        var norms = new float[Rows];
        for (var i = 0; i < Rows; i++)
        {
            var row = Row(i);
            var norm = 0.0f;
            for (var j = 0; j < row.Length; j++)
                norm += row[j] * row[j];
            norms[i] = MathF.Sqrt(norm);
        }
        return norms;
    }
}
