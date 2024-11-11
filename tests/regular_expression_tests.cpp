//
// Created by nik on 6/9/2024.
//

#include <gtest/gtest.h>

#include "../library/nml/primitives/regular_expressions.h"

using namespace nml;

TEST(regular_expression_tests, is_match)
{
    ASSERT_EQ(Regex("$").find_matches("abcd", 0, nullptr), 4);
    ASSERT_EQ(Regex("^").find_matches("abcd", 0, nullptr), 0);
    ASSERT_EQ(Regex("x|^").find_matches("abcd", 0, nullptr), 0);
    ASSERT_EQ(Regex("x|$").find_matches("abcd", 0, nullptr), 4);
    ASSERT_EQ(Regex("x").find_matches("abcd", 0, nullptr), -1);
    ASSERT_EQ(Regex(".").find_matches("abcd", 0, nullptr), 1);
    ASSERT_EQ(Regex("^.*\\\\.*$").find_matches("c:\\Tools", 0, nullptr), 8);

    ASSERT_EQ(Regex::compile("\\").err(), Regex::ErrorCode::INVALID_METACHARACTER);
    ASSERT_EQ(Regex::compile("\\x").err(), Regex::ErrorCode::INVALID_METACHARACTER);
    ASSERT_EQ(Regex::compile("\\x1").err(), Regex::ErrorCode::INVALID_METACHARACTER);

    ASSERT_EQ(Regex("\\x20").find_matches(" ", 0, nullptr), 1);
    ASSERT_EQ(Regex("^.+$").find_matches("", 0, nullptr), -1);
    ASSERT_EQ(Regex("^(.+)$").find_matches("", 0, nullptr), -1);
    ASSERT_EQ(Regex("^([\\+-]?)([\\d]+)$").find_matches("+", 0, nullptr), -1);
    ASSERT_EQ(Regex("^([\\+-]?)([\\d]+)$").find_matches("+27", 0, nullptr), 3);

    Regex::Match match;
    ASSERT_EQ(Regex("tel:\\+(\\d+[\\d-]+\\d)").find_matches("tel:+1-201-555-0123;a=b", 0, &match), 19);
    ASSERT_EQ(match.groups[0].length, 14);
    ASSERT_EQ(memcmp(match.groups[0].capture, "1-201-555-0123", match.groups[0].length), 0);

    ASSERT_EQ(Regex("[abc]").find_matches( "1c2", 0, nullptr), 2);
    ASSERT_EQ(Regex("[abc]").find_matches( "1C2", 0, nullptr), -1);
    ASSERT_EQ(Regex("[.2]").find_matches( "1C2", 0, nullptr), 1);
    ASSERT_EQ(Regex("[\\S]+").find_matches( "ab cd", 0, nullptr), 2);
    ASSERT_EQ(Regex("[\\S]+\\s+[tyc]*").find_matches( "ab cd", 0, nullptr), 4);
    ASSERT_EQ(Regex("[\\d]").find_matches( "ab cd", 0, nullptr), -1);
    ASSERT_EQ(Regex("[^\\d]").find_matches( "ab cd", 0, nullptr), 1);
    ASSERT_EQ(Regex("[^\\d]+").find_matches( "abc123", 0, nullptr), 3);
    ASSERT_EQ(Regex("[1-5]+").find_matches( "123456789", 0, nullptr), 5);
    ASSERT_EQ(Regex("[1-5a-c]+").find_matches( "123abcdef", 0, nullptr), 6);
    ASSERT_EQ(Regex("[1-5a-]+").find_matches( "123abcdef", 0, nullptr), 4);
    ASSERT_EQ(Regex("[1-5a-]+").find_matches( "123a--2oo", 0, nullptr), 7);
    ASSERT_EQ(Regex("[htps]+://").find_matches( "https://", 0, nullptr), 8);
    ASSERT_EQ(Regex("[^\\s]+").find_matches( "abc def", 0, nullptr), 3);
    ASSERT_EQ(Regex("[^fc]+").find_matches( "abc def", 0, nullptr), 2);
    ASSERT_EQ(Regex("[^d\\sf]+").find_matches( "abc def", 0, nullptr), 3);

    ASSERT_EQ(Regex("fo").find_matches( "foo", 0, nullptr), 2);
    ASSERT_EQ(Regex(".+").find_matches("foo", 0, nullptr), 3);
    ASSERT_EQ(Regex(".+k").find_matches("fooklmn", 0, nullptr), 4);
    ASSERT_EQ(Regex(".+k.").find_matches("fooklmn", 0, nullptr), 5);
    ASSERT_EQ(Regex("p+").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex("ok").find_matches("fooklmn", 0, nullptr), 4);
    ASSERT_EQ(Regex("lmno").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex("mn.").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex("o").find_matches("fooklmn", 0, nullptr), 2);
    ASSERT_EQ(Regex("^o").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex("^").find_matches("fooklmn", 0, nullptr), 0);
    ASSERT_EQ(Regex("n$").find_matches("fooklmn", 0, nullptr), 7);
    ASSERT_EQ(Regex("n$k").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex("l$").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex(".$").find_matches("fooklmn", 0, nullptr), 7);
    ASSERT_EQ(Regex("a?").find_matches("fooklmn", 0, nullptr), 0);
    ASSERT_EQ(Regex("^a*CONTROL").find_matches("CONTROL", 0, nullptr), 7);
    ASSERT_EQ(Regex("^[a]*CONTROL").find_matches("CONTROL", 0, nullptr), 7);
    ASSERT_EQ(Regex("^(a*)CONTROL").find_matches("CONTROL", 0, nullptr), 7);
    ASSERT_EQ(Regex("^(a*)?CONTROL").find_matches("CONTROL", 0, nullptr), 7);

    ASSERT_EQ(Regex("()+").find_matches("fooklmn", 0, nullptr), -1);
    ASSERT_EQ(Regex("\\x20").find_matches("_ J", 0, nullptr), 2);
    ASSERT_EQ(Regex("\\x4A").find_matches("_ J", 0, nullptr), 3);
    ASSERT_EQ(Regex("\\d+").find_matches("abc123def", 0, nullptr), 6);

    ASSERT_EQ(Regex("klz?mn").find_matches("fooklmn", 0, nullptr), 7);
    ASSERT_EQ(Regex("fa?b").find_matches("fooklmn", 0, nullptr), -1);

    ASSERT_EQ(Regex("^(te)").find_matches( "tenacity subdues all", 0, nullptr), 2);
    ASSERT_EQ(Regex("(bc)").find_matches("abcdef", 0, nullptr), 3);
    ASSERT_EQ(Regex(".(d.)").find_matches("abcdef", 0, nullptr), 5);
    ASSERT_EQ(Regex(".(d.)\\)?").find_matches( "abcdef", 0, &match), 5);
    ASSERT_EQ(match.groups[0].length,  2);
    ASSERT_EQ(memcmp(match.groups[0].capture, "de", 2), 0);
    ASSERT_EQ(Regex("(.+)").find_matches("123", 0, nullptr), 3);
    ASSERT_EQ(Regex("(2.+)").find_matches("123", 0, &match), 3);
    ASSERT_EQ(match.groups[0].length, 2);
    ASSERT_EQ(memcmp(match.groups[0].capture, "23", 2), 0);
    ASSERT_EQ(Regex("(.+2)").find_matches("123", 0, &match), 2);
    ASSERT_EQ(match.groups[0].length, 2);
    ASSERT_EQ(memcmp(match.groups[0].capture, "12", 2), 0);
    ASSERT_EQ(Regex("(.*(2.))").find_matches("123", 0, nullptr), 3);
    ASSERT_EQ(Regex("(.)(.)").find_matches("123", 0, nullptr), 2);
    ASSERT_EQ(Regex("(\\d+)\\s+(\\S+)").find_matches("12 hi", 0, nullptr), 5);
    ASSERT_EQ(Regex("ab(cd)+ef").find_matches("abcdcdef", 0, nullptr), 8);
    ASSERT_EQ(Regex("ab(cd)*ef").find_matches("abcdcdef", 0, nullptr), 8);
    ASSERT_EQ(Regex("ab(cd)+?ef").find_matches("abcdcdef", 0, nullptr), 8);
    ASSERT_EQ(Regex("ab(cd)+?.").find_matches("abcdcdef", 0, nullptr), 5);
    ASSERT_EQ(Regex("ab(cd)?").find_matches("abcdcdef", 0, nullptr), 4);
    ASSERT_EQ(Regex("(.+/\\d+\\.\\d+)\\.jpg$").find_matches("/foo/bar/12.34.jpg", 0, nullptr), 18);
    ASSERT_EQ(Regex("(ab|cd).*\\.(xx|yy)").find_matches("ab.yy", 0, nullptr), 5);
    ASSERT_EQ(Regex(".*a").find_matches("abcdef", 0, nullptr), 1);
    ASSERT_EQ(Regex("(.+)c").find_matches("abcdef", 0, nullptr), 3);
    ASSERT_EQ(Regex("\\n").find_matches("abc\ndef", 0, nullptr), 4);
    ASSERT_EQ(Regex("b.\\s*\\n").find_matches("aa\r\nbb\r\ncc\r\n\r\n", 0, nullptr), 8);


    ASSERT_EQ(Regex(".+c").find_matches("abcabc", 0, nullptr), 6);
    ASSERT_EQ(Regex(".+?c").find_matches("abcabc", 0, nullptr), 3);
    ASSERT_EQ(Regex(".*?c").find_matches("abcabc", 0, nullptr), 3);
    ASSERT_EQ(Regex(".*c").find_matches("abcabc", 0, nullptr), 6);
    ASSERT_EQ(Regex("bc.d?k?b+").find_matches("abcabc", 0, nullptr), 5);

    ASSERT_EQ(Regex("|").find_matches( "abc", 0, nullptr), 0);
    ASSERT_EQ(Regex("|.").find_matches( "abc", 0, nullptr), 1);
    ASSERT_EQ(Regex("x|y|b").find_matches( "abc", 0, nullptr), 2);
    ASSERT_EQ(Regex("k(xx|yy)|ca").find_matches( "abcabc", 0, nullptr), 4);
    ASSERT_EQ(Regex("k(xx|yy)|ca|bc").find_matches( "abcabc", 0, nullptr), 3);
    ASSERT_EQ(Regex("(|.c)").find_matches( "abc", 0, &match), 3);
    ASSERT_EQ(match.groups[0].length, 2);
    ASSERT_EQ(memcmp(match.groups[0].capture, "bc", 2), 0);
    ASSERT_EQ(Regex("a|b|c").find_matches( "a", 0, nullptr), 1);
    ASSERT_EQ(Regex("a|b|c").find_matches( "b", 0, nullptr), 1);
    ASSERT_EQ(Regex("a|b|c").find_matches( "c", 0, nullptr), 1);
    ASSERT_EQ(Regex("a|b|c").find_matches("d", 0, nullptr), -1);

    ASSERT_EQ(Regex("^.*c.?$").find_matches("abc", 0, nullptr), 3);
//    ASSERT_EQ(Regex(("^.*C.?$", "abc", 0, nullptr), 3);
    ASSERT_EQ(Regex("bk?").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("b(k?)").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("b[k-z]*").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("ab(k|z|y)*").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("[b-z].*").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("(b|z|u).*").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("ab(k|z|y)?").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex(".*").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex(".*$").find_matches("ab", 0, nullptr), 2);
    ASSERT_EQ(Regex("a+$").find_matches("aa", 0, nullptr), 2);
    ASSERT_EQ(Regex("a*$").find_matches("aa", 0, nullptr), 2);
    ASSERT_EQ(Regex( "a+$").find_matches("Xaa", 0, nullptr), 3);
    ASSERT_EQ(Regex( "a*$").find_matches("Xaa", 0, nullptr), 3);

    ASSERT_GT(Regex( "^hello$", Regex::Option::MULTI_LINE | Regex::Option::INSENSITIVE)
        .find_matches("Hello\nWorld", 0, nullptr), 0);

//    ASSERT_GT(Regex("^\\s*(\\S+)\\s+(\\S+)\\s+HtTP/(\\d)\\.(\\d)", Regex::Option::INSENSITIVE | Regex::Option::SINGLE_LINE)
//            .find_matches(" GET /index.html HTTP/1.0\r\n\r\n", 0, &match), 0);
//
//    ASSERT_EQ(match.groups[1].length, 11);
//    ASSERT_EQ(memcmp(match.groups[1].capture, "/index.html", match.groups[1].length), 0);

//    match.print();
//
//
//    ASSERT_TRUE(match.is_match());
}

TEST(regular_expression_tests, bigger)
{
    const char* string = "<img src=\"HTTPS://google.cOm/\"/> "
            "  <a href=\"http://en.wikipedia.org/wiki/Eigenvalue_algorithm\">Eigenvalues</a>";

    auto regex_result = Regex::compile(R"(((https?://)[^\s/'"<>]+/?[^\s'"<>]*))", Regex::Option::INSENSITIVE);

    ASSERT_TRUE(regex_result.is_ok());

    auto regex = regex_result.ok();

    auto matches = regex.matches(string);

    while (matches.has_next())
    {
        auto current = matches.next();

        printf("Found URL: %.*s\n", current.length, current.match);
    }

    ASSERT_EQ(matches.match_count(), 2);
}