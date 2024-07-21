# Disco.NET

Recommendations for .NET using collaborative filtering

- Supports user-based and item-based recommendations
- Works with explicit and implicit feedback
- Uses high-performance matrix factorization

[![Build Status](https://github.com/ankane/disco-dotnet/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/disco-dotnet/actions)

## Installation

Run

```sh
dotnet add package DiscoRec
```

## Getting Started

Import the library

```csharp
using DiscoRec;
```

Prep your data in the format `userId, itemId, value`

```csharp
var data = new Dataset<string, string>();
data.Add("user_a", "item_a", 5.0f);
data.Add("user_a", "item_b", 3.5f);
data.Add("user_b", "item_a", 4.0f);
```

IDs can be integers or strings

```csharp
data.Add(1, "item_a", 5.0f);
```

If users rate items directly, this is known as explicit feedback. Fit the recommender with:

```csharp
var recommender = Recommender<string, string>.FitExplicit(data);
```

If users don’t rate items directly (for instance, they’re purchasing items or reading posts), this is known as implicit feedback. Leave out the rating.

```csharp
var data = new Dataset<string, string>();
data.Add("user_a", "item_a");
data.Add("user_a", "item_b");
data.Add("user_b", "item_a");

var recommender = Recommender<string, string>.FitImplicit(data);
```

> Each `userId`/`itemId` combination should only appear once

Get user-based recommendations - “users like you also liked”

```csharp
recommender.UserRecs(userId, 5);
```

Get item-based recommendations - “users who liked this item also liked”

```csharp
recommender.ItemRecs(itemId, 5);
```

Get predicted ratings for a specific user and item

```csharp
recommender.Predict(userId, itemId);
```

Get similar users

```csharp
recommender.SimilarUsers(userId, 5);
```

## Examples

### MovieLens

Load the data

```csharp
var data = await Data.LoadMovieLens();
```

Create a recommender

```csharp
var recommender = Recommender<int, string>.FitExplicit(data, new RecommenderOptions { Factors = 20 });
```

Get similar movies

```csharp
var recs = recommender.ItemRecs("Star Wars (1977)");
foreach (var rec in recs)
    Console.WriteLine("{0}: {1}", rec.Id, rec.Score);
```

## Storing Recommendations

Save recommendations to your database.

Alternatively, you can store only the factors and use a library like [pgvector-dotnet](https://github.com/pgvector/pgvector-dotnet). See an [example](https://github.com/pgvector/pgvector-dotnet/blob/master/examples/Disco/Program.cs).

## Algorithms

Disco uses high-performance matrix factorization.

- For explicit feedback, it uses [stochastic gradient descent](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf_journal.pdf)
- For implicit feedback, it uses [coordinate descent](https://www.csie.ntu.edu.tw/~cjlin/papers/one-class-mf/biased-mf-sdm-with-supp.pdf)

Specify the number of factors and iterations

```csharp
var recommender = Recommender<string, string>.FitExplicit(data, new RecommenderOptions { Factors = 8, Iterations = 20 });
```

## Validation

Pass a validation set

```csharp
var recommender = Recommender<string, string>.FitExplicit(trainSet, validSet);
```

## Cold Start

Collaborative filtering suffers from the [cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)). It’s unable to make good recommendations without data on a user or item, which is problematic for new users and items.

```csharp
recommender.UserRecs(newUserId, 5);
```

There are a number of ways to deal with this, but here are some common ones:

- For user-based recommendations, show new users the most popular items
- For item-based recommendations, make content-based recommendations

## Reference

Get ids

```csharp
recommender.UserIds();
recommender.ItemIds();
```

Get the global mean

```csharp
recommender.GlobalMean();
```

Get factors

```csharp
recommender.UserFactors(userId);
recommender.ItemFactors(itemId);
```

## History

View the [changelog](https://github.com/ankane/disco-dotnet/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/disco-dotnet/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/disco-dotnet/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/disco-dotnet.git
cd disco-dotnet
dotnet test
```
