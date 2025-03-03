package edu.emory.mathcs.nlp.component.tokenizer;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;
import edu.emory.mathcs.nlp.component.tokenizer.token.Token;

import java.util.List;
import java.util.Scanner;

public class TokenizerUtil {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("分词工具已启动，输入q退出");

        while (true) {
            System.out.print("请输入文本 > ");
            String input = scanner.nextLine().trim();

            if ("q".equalsIgnoreCase(input)) {
                System.out.println("已退出分词工具");
                break;
            }

            String result = tokenizeWithSlashes(input);
            System.out.println("分词结果: " + result);
        }
        scanner.close();
    }

    private static String tokenizeWithSlashes(String input) {
        if (containsChinese(input)) {
            List<Term> terms = HanLP.segment(input); // 中文分词 <button class="citation-flag" data-index="3"><button class="citation-flag" data-index="8">
            return String.join("/", terms.stream().map(t -> t.word).toArray(String[]::new));
        } else {
            Tokenizer tokenizer = new EnglishTokenizer(); // 英文分词 <button class="citation-flag" data-index="1">
            List<Token> tokens = tokenizer.tokenize(input);
            return String.join("/", tokens.stream().map(Token::getWordForm).toArray(String[]::new));
        }
    }

    private static boolean containsChinese(String input) {
        return input.codePoints().anyMatch(c -> c >= 0x4E00 && c <= 0x9FFF); // Unicode中文范围检测
    }
}