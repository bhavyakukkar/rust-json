# rust-json

A Rust rewrite of the first commit of [Tsoding's JSON Parser in Haskell](https://github.com/tsoding/haskell-json/blob/bafd97d96b792edd3e170525a7944b9f01de7e34/Main.hs),
made by him in this [wonderful & educative stream](https://www.youtube.com/watch?v=N9RUqGYuGfw).

I decided against making a 1-1 rewrite including implementing generic definitions of Haskell
type-classes like `Functors` and `Applicative Functors`, as introducing such a workflow in a
language that does not provide it would account for lots of boilerplate & unreadable code.

The `main` module behaves as a library that you can copy and use in your own project.

**You can also use this as a JSON pretty-printer** like so:

### build
```sh
rustc --edition 2021 ./main.rs -o json-parser
```

### usage
```sh
./json-parser <file.json> [<indent-size>]

# example
./json-parser package.json 2
```

### test
```sh
rustc --test --edition 2021 ./main.rs -o tests
./tests
```
