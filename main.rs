// module Main where
mod main {

    // Some std imports
    use std::fmt;

    // Firstly, some Haskell-like stuff
    #[derive(Clone, Copy)]
    pub struct LazyCopy<A>(fn() -> A);
    impl<A: fmt::Debug> fmt::Debug for LazyCopy<A> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "LazyCopy(<closure>)")
        }
    }
    struct LazyClone<A>(Box<dyn Fn() -> A>);
    impl<A: fmt::Debug> fmt::Debug for LazyClone<A> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "LazyClone(<closure>)")
        }
    }
    pub struct Lazy<A>(pub Box<dyn FnOnce() -> A>);
    impl<A: fmt::Debug> fmt::Debug for Lazy<A> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Lazy(<closure>)")
        }
    }
    impl<A> Lazy<A> {
        fn eval(self) -> A {
            self.0()
        }
    }

    // data JsonValue
    //   = JsonNull
    //   | JsonBool Bool
    //   | JsonNumber Integer -- NOTE: no support for floats
    //   | JsonString String
    //   | JsonArray [JsonValue]
    //   | JsonObject [(String, JsonValue)]
    //   deriving (Show, Eq)
    #[allow(clippy::enum_variant_names)]
    #[derive(Debug, Clone, PartialEq)]
    pub enum JsonValue {
        JsonNull,
        JsonBool(bool),
        JsonNumber(usize),
        JsonString(String),
        JsonArray(Vec<JsonValue>),
        JsonObject(Vec<(String, JsonValue)>),
    }
    use JsonValue::*;
    impl JsonValue {
        pub fn pretty_printer(self, indent_size: usize) -> JsonPrettyPrinter {
            JsonPrettyPrinter(self, 0, indent_size)
        }
    }
    pub struct JsonPrettyPrinter(JsonValue, usize, usize);
    impl fmt::Display for JsonPrettyPrinter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            use fmt::Write;
            let indent = |n| " ".repeat(self.2).repeat(n);
            match &self.0 {
                JsonNull => write!(f, "null"),

                JsonBool(true) => write!(f, "true"),
                JsonBool(false) => write!(f, "false"),

                JsonNumber(n) => write!(f, "{}", n),
                JsonString(s) => write!(f, "\"{}\"", s),
                JsonArray(a) => {
                    let a = a.iter();
                    write!(
                        f,
                        "[\n{}{i}]",
                        a.fold(String::new(), |mut output, item| {
                            let _ = writeln!(
                                output,
                                "{i}{},",
                                JsonPrettyPrinter(item.clone(), self.1 + 1, self.2),
                                i = indent(self.1 + 1),
                            );
                            output
                        }),
                        i = indent(self.1),
                    )
                }
                JsonObject(o) => {
                    let o = o.iter();
                    write!(
                        f,
                        "{{\n{}{i}}}",
                        o.fold(String::new(), |mut output, (key, value)| {
                            let _ = writeln!(
                                output,
                                "{i}\"{key}\": {},",
                                JsonPrettyPrinter(value.clone(), self.1 + 1, self.2),
                                i = indent(self.1 + 1),
                            );
                            output
                        }),
                        i = indent(self.1),
                    )
                }
            }
        }
    }

    // newtype Parser a = Parser
    //   { runParser :: String -> Maybe (String, a)
    //   }
    pub struct Parser<A>(Box<dyn FnOnce(String) -> Option<(String, A)>>);
    impl<A> Parser<A> {
        fn new(run_parser: Box<dyn FnOnce(String) -> Option<(String, A)>>) -> Self {
            Self(run_parser)
        }
        pub fn run_parser(self, input: String) -> Option<(String, A)> {
            self.0(input)
        }
    }
    impl<A: fmt::Debug> fmt::Debug for Parser<A> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Parser(<closure>)")
        }
    }

    // instance Functor Parser where
    //   fmap f (Parser p) =
    //     Parser $ \input -> do
    //       (input', x) <- p input
    //       Just (input', f x)
    impl<A: 'static> Parser<A> {
        fn fmap<B, F>(self, f: F) -> Parser<B>
        where
            B: 'static,
            F: FnOnce(A) -> B + 'static,
        {
            Parser::new(Box::new(move |input| {
                let (input, a) = self.0(input)?;
                Some((input, f(a)))
            }))
        }
    }

    // instance Applicative Parser where
    //   pure x = Parser $ \input -> Just (input, x)
    //   (Parser p1) <*> (Parser p2) =
    //     Parser $ \input -> do
    //       (input', f) <- p1 input
    //       (input'', a) <- p2 input'
    //       Just (input'', f a)
    impl<A: 'static> Parser<A> {
        fn pure(a: A) -> Self {
            Parser::new(Box::new(|input| Some((input, a))))
        }
    }
    impl<F> Parser<F> {
        fn seq<A: 'static, B: 'static>(self, other: Parser<A>) -> Parser<B>
        where
            F: FnOnce(A) -> B + 'static,
        {
            Parser::new(Box::new(|input| {
                let (input, f) = self.0(input)?;
                let (input, a) = other.0(input)?;
                Some((input, f(a)))
            }))
        }
    }

    // instance Alternative Parser where
    //   empty = Parser $ \_ -> Nothing
    //   (Parser p1) <|> (Parser p2) =
    //       Parser $ \input -> p1 input <|> p2 input
    impl<A> Parser<A>
    where
        A: fmt::Debug + 'static,
    {
        fn _empty() -> Self {
            Parser(Box::new(|_| None))
        }
        fn assoc(self, other: Self) -> Self {
            Parser::new(Box::new(|input| self.0(input.clone()).or(other.0(input))))
        }
    }

    // jsonNull :: Parser JsonValue
    // jsonNull = (\_ -> JsonNull) <$> stringP "null"
    pub fn json_null() -> Parser<JsonValue> {
        string_p("null").fmap(Box::new(|_| JsonNull))
    }

    // charP :: Char -> Parser Char
    // charP x = Parser f
    //   where
    //     f (y:ys)
    //       | y == x = Just (ys, x)
    //       | otherwise = Nothing
    //     f [] = Nothing
    pub fn char_p(x: char) -> Parser<char> {
        Parser::new(Box::new(move |input| {
            let mut chars = input.chars();
            match chars.next() {
                Some(y) if y == x => Some((chars.collect(), x)),
                _ => None,
            }
        }))
    }

    // stringP :: String -> Parser String
    // stringP = sequenceA . map charP
    pub fn string_p(input: &'static str) -> Parser<String> {
        sequence_a::<char, Vec<char>>(Box::new(input.chars().map(char_p)))
            .fmap(Box::new(String::from_iter))
    }
    fn sequence_a<T, V>(mut tra: Box<dyn Iterator<Item = Parser<T>>>) -> Parser<V>
    where
        T: 'static,
        V: FromIterator<T> + IntoIterator<Item = T> + 'static,
    {
        match tra.next() {
            Some(parser) => parser
                .fmap(Box::new(|inner: T| {
                    |old_it: V| -> V {
                        V::from_iter(V::from_iter([inner]).into_iter().chain(old_it))
                    }
                }))
                .seq(sequence_a(tra)),
            None => Parser::<V>::pure(V::from_iter([])),
        }
    }

    // jsonBool :: Parser JsonValue
    // jsonBool = f <$> (stringP "true" <|> stringP "false")
    //    where f "true"  = JsonBool True
    //          f "false" = JsonBool False
    //          -- This should never happen
    //          f _       = undefined
    pub fn json_bool() -> Parser<JsonValue> {
        let f = |input: String| match input.as_str() {
            "true" => JsonBool(true),
            "false" => JsonBool(false),
            _ => panic!("This should never happen"),
        };
        string_p("true").assoc(string_p("false")).fmap(f)
    }

    // spanP :: (Char -> Bool) -> Parser String
    // spanP f =
    //   Parser $ \input ->
    //     let (token, rest) = span f input
    //      in Just (rest, token)
    pub fn span_p<F>(predicate: F) -> Parser<String>
    where
        F: FnMut(&char) -> bool + Clone + 'static,
    {
        Parser::new(Box::new(|input| {
            let (token, rest) = span(predicate, input.chars().collect::<Vec<char>>());
            Some((String::from_iter(rest), String::from_iter(token)))
        }))
    }
    fn span<T, V, F>(predicate: F, list: V) -> (V, V)
    where
        V: FromIterator<T> + IntoIterator<Item = T> + Clone,
        F: FnMut(&T) -> bool + Clone,
    {
        (
            V::from_iter(list.clone().into_iter().take_while(predicate.clone())),
            V::from_iter(list.into_iter().skip_while(predicate)),
        )
    }

    // notNull :: Parser [a] -> Parser [a]
    // notNull (Parser p) =
    //   Parser $ \input -> do
    //     (input', xs) <- p input
    //     if null xs
    //       then Nothing
    //       else Just (input', xs)
    pub fn not_null<T, V>(p: Parser<V>) -> Parser<V>
    where
        V: FromIterator<T> + IntoIterator<Item = T> + Clone + 'static,
    {
        Parser::new(Box::new(|input| {
            let (input, xs) = p.0(input)?;
            match xs.clone().into_iter().next().is_none() {
                true => None,
                false => Some((input, xs)),
            }
        }))
    }

    // jsonNumber :: Parser JsonValue
    // jsonNumber = f <$> notNull (spanP isDigit)
    //     where f ds = JsonNumber $ read ds
    pub fn json_number() -> Parser<JsonValue> {
        let f = |ds| JsonNumber(str::parse(&String::from_iter(ds)).expect("This should not fail"));
        not_null(span_p(|c| c.is_ascii_digit()).fmap(|s| s.chars().collect::<Vec<char>>())).fmap(f)
    }

    // -- NOTE: no escape support
    // stringLiteral :: Parser String
    // stringLiteral = charP '"' *> spanP (/= '"') <* charP '"'
    pub fn string_literal() -> Parser<String> {
        char_p('"')
            .seq_right(span_p(|c| *c != '"'))
            .seq_left(char_p('"'))
    }
    impl<A> Parser<A>
    where
        A: fmt::Debug + 'static,
    {
        fn seq_right<B>(self, other: Parser<B>) -> Parser<B>
        where
            B: fmt::Debug + 'static,
        {
            Parser::new(Box::new(|input| match self.0(input) {
                Some((input, _)) => other.0(input),
                None => None,
            }))
        }
        fn seq_left<B: 'static>(self, other: Parser<B>) -> Self {
            Parser::new(Box::new(|input| match self.0(input) {
                Some((input, a)) => other.0(input).map(|(input, _)| (input, a)),
                None => None,
            }))
        }
    }

    // jsonString :: Parser JsonValue
    // jsonString = JsonString <$> stringLiteral
    pub fn json_string() -> Parser<JsonValue> {
        string_literal().fmap(JsonString)
    }

    // ws :: Parser String
    // ws = spanP isSpace
    pub fn ws() -> Parser<String> {
        span_p(|c| c.is_whitespace())
    }

    // sepBy :: Parser a -> Parser b -> Parser [b]
    // sepBy sep element = (:) <$> element <*> many (sep *> element) <|> pure []
    pub fn sep_by<A, B, V>(
        sep: LazyCopy<Parser<A>>,
        element: LazyCopy<Parser<B>>,
    ) -> Lazy<Parser<V>>
    where
        A: fmt::Debug + 'static,
        B: fmt::Debug + Clone + 'static,
        V: fmt::Debug + IntoIterator<Item = B> + FromIterator<B> + 'static,
    {
        Lazy(Box::new(move || {
            element.0()
                .fmap(|b| move |old_it: V| V::from_iter([b].into_iter().chain(old_it)))
                .seq(LazyClone(Box::new(move || sep.0().seq_right(element.0()))).many())
                .assoc(Parser::pure(V::from_iter([])))
        }))
    }
    impl<A> LazyClone<Parser<A>>
    where
        A: fmt::Debug + 'static,
    {
        fn many<V>(self) -> Parser<V>
        where
            V: FromIterator<A>,
        {
            Parser::new(Box::new(move |mut input| {
                let mut it = Vec::new();
                while let Some((rest, a)) = self.0().0(input.clone()) {
                    it.push(a);
                    input = rest;
                }
                Some((input, V::from_iter(it)))
            }))
        }
    }

    // jsonArray :: Parser JsonValue
    // jsonArray = JsonArray <$> (charP '[' *> ws *>
    //                            elements
    //                            <* ws <* charP ']')
    //   where
    //     elements = sepBy (ws *> charP ',' <* ws) jsonValue
    pub fn json_array() -> Parser<JsonValue> {
        let elements: Lazy<Parser<Vec<JsonValue>>> = sep_by(
            LazyCopy(|| ws().seq_right(char_p(',')).seq_left(ws())),
            LazyCopy(json_value),
        );
        char_p('[')
            .seq_right(ws())
            .seq_right_lazy(elements)
            .seq_left(ws())
            .seq_left(char_p(']'))
            .fmap(JsonArray)
    }
    impl<A: 'static> Parser<A> {
        fn seq_right_lazy<B>(self, other: Lazy<Parser<B>>) -> Parser<B>
        where
            B: fmt::Debug + 'static,
        {
            Parser::new(Box::new(|input| match self.0(input) {
                Some((input, _)) => other.eval().0(input),
                None => None,
            }))
        }
    }

    // jsonObject :: Parser JsonValue
    // jsonObject =
    //   JsonObject <$> (charP '{' *> ws *> sepBy (ws *> charP ',' <* ws) pair <* ws <* charP '}')
    //   where
    //     pair =
    //       (\key _ value -> (key, value)) <$> stringLiteral <*>
    //       (ws *> charP ':' <* ws) <*>
    //       jsonValue
    pub fn json_object() -> Parser<JsonValue> {
        let pair = LazyCopy(|| {
            string_literal()
                .fmap(|key| move |_| move |value| (key, value))
                .seq(ws().seq_right(char_p(':')).seq_left(ws()))
                .seq(json_value())
        });
        char_p('{')
            .seq_right(ws())
            .seq_right_lazy(sep_by(
                LazyCopy(|| ws().seq_right(char_p(',')).seq_left(ws())),
                pair,
            ))
            .seq_left(ws())
            .seq_left(char_p('}'))
            .fmap(JsonObject)
    }

    // jsonValue :: Parser JsonValue
    // jsonValue = jsonNull <|> jsonBool <|> jsonNumber <|> jsonString <|> jsonArray <|> jsonObject
    pub fn json_value() -> Parser<JsonValue> {
        json_null()
            .assoc(json_bool())
            .assoc(json_number())
            .assoc(json_string())
            .assoc(json_array())
            .assoc(json_object())
    }
}

fn main() {
    use main::json_value;
    use std::{env, fs, process};

    let exit_with_help = |exec_name| {
        println!("Usage: {} <file.json> [<indent-size>]", exec_name);
        process::exit(1);
    };

    let mut args = env::args();
    let exec_name = args
        .next()
        .ok_or_else(|| {
            eprintln!("Expecting at least 1 argument");
            exit_with_help("json-parser".to_owned());
        })
        .unwrap();

    let file_name = args
        .next()
        .ok_or_else(|| {
            eprintln!("Filename not found.");
            exit_with_help(exec_name);
        })
        .unwrap();

    let indent_size = args
        .next()
        .map(|s| str::parse::<usize>(&s).unwrap_or(2))
        .unwrap_or(2);

    let json = json_value()
        .run_parser(fs::read_to_string(file_name).expect("Expecting valid file"))
        .ok_or_else(|| {
            eprintln!("Failed to parse, invalid JSON.");
            process::exit(1);
        })
        .unwrap();

    println!(
        "Parsed:\n{}\n\nRemaining:\n{}",
        json.1.pretty_printer(indent_size),
        json.0
    );
}

// Tests
#[cfg(test)]
mod tests {
    use crate::main::{
        json_array, json_bool, json_null, json_number, json_object, json_string, json_value,
        JsonValue,
    };
    use JsonValue::*;

    // Null - Passing
    #[test]
    fn null_pass_01() {
        assert_eq!(
            json_null().run_parser("null".to_owned()),
            Some(("".to_owned(), JsonNull))
        );
    }
    #[test]
    fn null_pass_02() {
        assert_eq!(
            json_null().run_parser("nullnullnull".to_owned()),
            Some(("nullnull".to_owned(), JsonNull))
        );
    }

    // Null - Failing
    #[test]
    fn null_fail_01() {
        assert_eq!(json_null().run_parser("".to_owned()), None);
    }
    #[test]
    fn null_fail_02() {
        assert_eq!(json_null().run_parser("nl".to_owned()), None);
    }
    #[test]
    fn null_fail_03() {
        assert_eq!(json_null().run_parser("nnull".to_owned()), None);
    }

    // Bool - Passing
    #[test]
    fn bool_pass_01() {
        assert_eq!(
            json_bool().run_parser("true".to_owned()),
            Some(("".to_owned(), JsonBool(true)))
        );
    }
    #[test]
    fn bool_pass_02() {
        assert_eq!(
            json_bool().run_parser("false".to_owned()),
            Some(("".to_owned(), JsonBool(false)))
        );
    }
    #[test]
    fn bool_pass_03() {
        assert_eq!(
            json_bool().run_parser("truefalse".to_owned()),
            Some(("false".to_owned(), JsonBool(true)))
        );
    }

    // Bool - Failing
    #[test]
    fn bool_fail_01() {
        assert_eq!(json_bool().run_parser("".to_owned()), None);
    }
    #[test]
    fn bool_fail_02() {
        assert_eq!(json_bool().run_parser("tr".to_owned()), None);
    }
    #[test]
    fn bool_fail_03() {
        assert_eq!(json_bool().run_parser("uefalse".to_owned()), None);
    }

    // Number - Passing
    #[test]
    fn number_pass_01() {
        assert_eq!(
            json_number().run_parser("1".to_owned()),
            Some(("".to_owned(), JsonNumber(1)))
        );
    }
    #[test]
    fn number_pass_02() {
        assert_eq!(
            json_number().run_parser("126392".to_owned()),
            Some(("".to_owned(), JsonNumber(126392)))
        );
    }
    #[test]
    fn number_pass_03() {
        assert_eq!(
            json_number().run_parser("69420hey".to_owned()),
            Some((("hey".to_owned()), JsonNumber(69420)))
        );
    }

    // Number - Failing
    #[test]
    fn number_fail_01() {
        assert_eq!(json_number().run_parser("".to_owned()), None);
    }
    #[test]
    fn number_fail_02() {
        assert_eq!(json_number().run_parser("aa1".to_owned()), None);
    }
    #[test]
    fn number_fail_03() {
        assert_eq!(json_number().run_parser("a1aa".to_owned()), None);
    }

    // String - Passing
    #[test]
    fn string_pass_01() {
        assert_eq!(
            json_string().run_parser("\"hey\"".to_owned()),
            Some(("".to_owned(), JsonString("hey".to_owned())))
        );
    }
    #[test]
    fn string_pass_02() {
        assert_eq!(
            json_string().run_parser("\"\"".to_owned()),
            Some(("".to_owned(), JsonString("".to_owned())))
        );
    }
    #[test]
    fn string_pass_03() {
        assert_eq!(
            json_string().run_parser("\"Hello, World! 42\",23".to_owned()),
            Some((",23".to_owned(), JsonString("Hello, World! 42".to_owned())))
        );
    }

    // String - Failing
    #[test]
    fn string_fail_01() {
        assert_eq!(json_string().run_parser("".to_owned()), None);
    }
    #[test]
    fn string_fail_02() {
        assert_eq!(json_string().run_parser("\"hey".to_owned()), None);
    }
    #[test]
    fn string_fail_03() {
        assert_eq!(json_string().run_parser("f\"hey".to_owned()), None);
    }

    // Array - Passing
    #[test]
    fn array_pass_01() {
        assert_eq!(
            json_array().run_parser("[]".to_owned()),
            Some(("".to_owned(), JsonArray(vec![])))
        );
    }
    #[test]
    fn array_pass_02() {
        assert_eq!(
            json_array().run_parser("[1]".to_owned()),
            Some(("".to_owned(), JsonArray(vec![JsonNumber(1)])))
        );
    }
    #[test]
    fn array_pass_03() {
        assert_eq!(
            json_array().run_parser("[ [1, true ,false ]  , \"Hello, World!\" ]".to_owned()),
            Some((
                "".to_owned(),
                JsonArray(vec![
                    JsonArray(vec![JsonNumber(1), JsonBool(true), JsonBool(false)]),
                    JsonString("Hello, World!".to_owned())
                ])
            ))
        );
    }
    #[test]
    fn array_pass_04() {
        assert_eq!(
            json_array().run_parser("[ null,{\"a\": 2}],123".to_owned()),
            Some((
                ",123".to_owned(),
                JsonArray(vec![
                    JsonNull,
                    JsonObject(vec![("a".to_owned(), JsonNumber(2))])
                ])
            ))
        );
    }

    // Array - Failing
    #[test]
    fn array_fail_01() {
        assert_eq!(json_array().run_parser("".to_owned()), None);
    }
    #[test]
    fn array_fail_02() {
        assert_eq!(json_array().run_parser("[".to_owned()), None);
    }
    #[test]
    fn array_fail_03() {
        assert_eq!(json_array().run_parser("14,[1,3]".to_owned()), None);
    }
    #[test]
    fn array_fail_04() {
        assert_eq!(json_array().run_parser("[ \"hey \" , 3,]".to_owned()), None);
    }
    #[test]
    fn array_fail_05() {
        assert_eq!(json_array().run_parser("[,,2]".to_owned()), None);
    }

    // Object - Passing
    #[test]
    fn object_pass_01() {
        assert_eq!(
            json_object().run_parser("{}".to_owned()),
            Some(("".to_owned(), JsonObject(vec![])))
        );
    }
    #[test]
    fn object_pass_02() {
        assert_eq!(
            json_object().run_parser("{\"one\":1}".to_owned()),
            Some((
                "".to_owned(),
                JsonObject(vec![("one".to_owned(), JsonNumber(1))])
            ))
        );
    }
    #[test]
    fn object_pass_03() {
        assert_eq!(
            json_object()
                .run_parser("{\"44\": \"hello\",\"4..5\"  :  {\"e\":5, \"\": \"\"}}".to_owned()),
            Some((
                "".to_owned(),
                JsonObject(vec![
                    ("44".to_owned(), JsonString("hello".to_owned())),
                    (
                        "4..5".to_owned(),
                        JsonObject(vec![
                            ("e".to_owned(), JsonNumber(5)),
                            ("".to_owned(), JsonString("".to_owned()))
                        ])
                    )
                ])
            ))
        );
    }
    #[test]
    fn object_pass_04() {
        assert_eq!(
            json_object().run_parser("{\"k\" :[null, true] },[12,3] ".to_owned()),
            Some((
                ",[12,3] ".to_owned(),
                JsonObject(vec![(
                    "k".to_owned(),
                    JsonArray(vec![JsonNull, JsonBool(true)])
                )])
            ))
        );
    }

    // Object - Failing
    #[test]
    fn object_fail_01() {
        assert_eq!(json_object().run_parser("".to_owned()), None);
    }
    #[test]
    fn object_fail_02() {
        assert_eq!(json_object().run_parser("{".to_owned()), None);
    }
    #[test]
    fn object_fail_03() {
        assert_eq!(json_object().run_parser("14,{1,3}".to_owned()), None);
    }
    #[test]
    fn object_fail_04() {
        assert_eq!(
            json_object().run_parser("{\"first\", \"second\": 2}".to_owned()),
            None
        );
    }
    #[test]
    fn object_fail_05() {
        assert_eq!(json_object().run_parser("{2:3}".to_owned()), None);
    }
    #[test]
    fn object_fail_06() {
        assert_eq!(json_object().run_parser("{\"h\":1,}".to_owned()), None);
    }
    #[test]
    fn object_fail_07() {
        assert_eq!(
            json_object().run_parser("{,, \"key\" :\"value\"}".to_owned()),
            None
        );
    }

    #[test]
    fn value_pass_01() {
        let json = "{ \
            \"hello\": [false, true, null, 42, \"foo\", [1, 2]],\
            \"world\": null \
            }2 ";
        let expected_ast = JsonObject(vec![
            (
                "hello".to_owned(),
                JsonArray(vec![
                    JsonBool(false),
                    JsonBool(true),
                    JsonNull,
                    JsonNumber(42),
                    JsonString("foo".to_owned()),
                    JsonArray(vec![JsonNumber(1), JsonNumber(2)]),
                ]),
            ),
            ("world".to_owned(), JsonNull),
        ]);

        assert_eq!(
            json_value().run_parser(json.to_owned()),
            Some(("2 ".to_owned(), expected_ast))
        );
    }

    #[test]
    fn value_pass_02() {
        let json = "[ \
            \"hello\", [false, true, null, 42, \"foo\", {\"a\": 1, \"b\": 2}],\
            \"world\", null \
            ], {}";
        let expected_ast = JsonArray(vec![
            JsonString("hello".to_owned()),
            JsonArray(vec![
                JsonBool(false),
                JsonBool(true),
                JsonNull,
                JsonNumber(42),
                JsonString("foo".to_owned()),
                JsonObject(vec![
                    ("a".to_owned(), JsonNumber(1)),
                    ("b".to_owned(), JsonNumber(2)),
                ]),
            ]),
            JsonString("world".to_owned()),
            JsonNull,
        ]);

        assert_eq!(
            json_value().run_parser(json.to_owned()),
            Some((", {}".to_owned(), expected_ast))
        );
    }
}
