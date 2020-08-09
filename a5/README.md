(a) (1 point) (written) In Assignment 4 we used 256-dimensional word embeddings (e word = 256), while in this assignment, it turns out that a character embedding size of 50 suffices (e char = 50). In 1-2 sentences, explain one reason why the embedding size used for character-level embeddings is typically lower than that used for word embeddings.

**ANSWER**: Characters are fewer and encode less information than words; it is then reasonable to use a smaller space for them

(b) (1 point) (written) Write down the total number of parameters in the character-based embedding model (Figure 2), then do the same for the word-based lookup embedding model (Figure 1). Write each answer as a single expression (though you may show working) in terms of e char , k, e word , V word (the size of the word-vocabulary in the lookup embedding model) and V char (the size of the character-vocabulary in the character-based embedding model).

Given that in our code, k = 5, V word ≈ 50, 000 and V char = 96, state which model has more parameters, and by what factor (e.g. twice as many? a thousand times as many?).


Char based emb: Vchar X E_char + e_word X (E_char X k) + f + 2 e_word^2 + 2 e_word
Word-based emb: e_word X V_word

Si k = 5, V_char = 96, V_word = 50K

Word -> 5 * 10^4 * 256 ~ 12.8M
Char -> 100 * 50 + 256 * 5* 50 + 256 + 2 * (256)^2 + 2 * 256 ~ 201K

Around 50 times less parameters!
