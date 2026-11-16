# TODO

- [ ] Decide whether GLiNER needs the fast tokenizer. Check whether `gliner` uses `offset_mapping`; if not, load the slow tokenizer and silence the "byte fallback" UserWarning from `convert_slow_tokenizer.py`. Context: `packages/backend/app/services/ner.py` (`_get_model`, `_count_tokens`).
- [ ] suggest improvements to golden set
  Suggested changes to the golden set, based on what the 20 current questions look like:
  **1. Fix the labels**
  - Gold pages are all page 1, so the metric rewards finding the title page. Label every page where the answer appears, not just the page that generated the question.
  - Store the supporting quote per page, so a label can be checked and reused.
  - Use document-level gold as the primary hit metric (any page from the right paper counts), and keep page-level as a secondary metric.
  **2. Fix the question mix**
  - Most questions name an author and ask about the paper's topic, which page 1 answers directly. Add questions that have no author names and are answered in the body (methods, results, numbers, definitions).
  - Add multi-paper questions, which need chunks from more than one document.
  - Add unanswerable questions, where the right behavior is to say the corpus doesn't cover it.
  - Tag every question with a type and report metrics per type, so you can see which kind a config handles well.
  **3. Size and noise**
  - Grow from 20 to about 100 to 200 questions, spread across all papers and across page positions. One question at 20 moves hit rate by 5 points, so differences between configs are mostly noise at this size.
  - Report a bootstrap confidence interval beside each metric.
  **4. Validation**
  - Check each question and answer for two things: the answer must come from the corpus, and the gold quote must support it. An LLM pass plus a manual sample of about 10% would catch most errors.
  **5. Process**
  - Split into a dev set for sweeps and a held-out set for the final choice, so the sweep cannot overfit the set it's scored on.
  - Version the golden set and store the version with each eval cycle, so metrics from different versions aren't compared.
  If you want, I can start with step 1: find every page for each of the current 20 questions that contains its gold answer, and show you the list before changing anything.

Alternate list:  
**What to change**

1. **Ground each question in a real page.** Sample pages across all documents and page positions, and send each page's text to the model. The gold page is then the page the question came from, not a guess.
2. **Generate one question per call.** Smaller, grounded calls are more reliable than one 20-item array.
3. **Verify before keeping.** A second pass checks that the question is answerable from the cited page and that the reference answer matches it. Drop anything that fails.
4. **Reject answerable-without-corpus questions.** Ask the model to answer without the page; if it can, the question doesn't test retrieval.
5. **Validate structure.** Check that the document exists, the page is in range and has text, and that questions aren't duplicated.
6. **Build multi-hop and negative items deliberately.** Pair pages from two papers for multi-hop questions, and add questions the corpus cannot answer.
7. **Record provenance.** Store the source page text (or its hash), the generator model, and the verifier model with each item.
8. **Balance the set.** Cap the share per document and per page position so the set isn't dominated by one kind of question.

Pending 6,7,8 from the above list

- [ ] Minor issue with corpus text: 
PDF page text is passed to the verifier LLM, so a crafted page could try to get a bad question marked as verified. Worth knowing if the corpus is untrusted.

- [ ] Tighten OCR code generation logic so the generated code checks output quality (empty or garbled text) before it's accepted.
