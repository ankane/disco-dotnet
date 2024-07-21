namespace DiscoRec;

using System;

internal class Matrix
{
    internal int Rows;
    private int Cols;
    private float[] Data;

    public Matrix(int rows, int cols, float[] data)
    {
        Rows = rows;
        Cols = cols;
        Data = data;
    }

    public float[] Row(int row)
    {
        var start = row * Cols;
        return Data[start..(start + Cols)];
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
