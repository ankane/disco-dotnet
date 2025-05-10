namespace DiscoRec;

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

public static class Data
{
    public static async Task<Dataset<int, string>> LoadMovieLens()
    {
        var itemPath = await DownloadFile(
            "ml-100k/u.item",
            "https://files.grouplens.org/datasets/movielens/ml-100k/u.item",
            "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701"
        );

        var dataPath = await DownloadFile(
            "ml-100k/u.data",
            "https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
            "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"
        );

        var movies = new Dictionary<string, string>(1682);
        foreach (var line in File.ReadLines(itemPath))
        {
            var row = line.Split("|");
            movies[row[0]] = row[1];
        }

        var data = new Dataset<int, string>(100000);
        foreach (var line in File.ReadLines(dataPath))
        {
            var row = line.Split("\t");
            data.Add(Int32.Parse(row[0]), movies[row[1]], Single.Parse(row[2]));
        }
        return data;
    }

    private static async Task<string> DownloadFile(string filename, string url, string fileHash)
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var dest = Path.Join(home, ".disco", filename);

        if (File.Exists(dest))
            return dest;

        Directory.CreateDirectory(Path.GetDirectoryName(dest)!);

        Console.WriteLine("Downloading data from {0}", url);
        var client = new HttpClient();
        using HttpResponseMessage response = await client.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var contents = await response.Content.ReadAsByteArrayAsync();

        var checksum = Convert.ToHexString(SHA256.HashData(contents)).ToLower();
        if (checksum != fileHash)
            throw new ArgumentException($"Bad checksum: {checksum}");

        File.WriteAllBytes(dest, contents);

        return dest;
    }
}
