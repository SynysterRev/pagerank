import os
import random
import re
import sys

from numpy import random as nprand

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, current_page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}
    visitable_pages = corpus[current_page]
    total_pages_corpus = len(corpus)
    random_choice_probability = (1 - damping_factor) / total_pages_corpus
    # no links
    if len(visitable_pages) == 0:
        for page in corpus:
            probability_distribution[page] = 1 / total_pages_corpus

    pages_probability = damping_factor / len(visitable_pages)
    for page in corpus:
        probability_distribution[page] = random_choice_probability + (pages_probability
                                                                      if page in visitable_pages else 0)
    probability_distribution = dict(sorted(probability_distribution.items()))
    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    corpus = dict(sorted(corpus.items()))
    estimated_page_rank = {}
    page = random.choice(list(corpus.keys()))
    all_pages = list(corpus.keys())
    for create_page in all_pages:
        estimated_page_rank[create_page] = 0
    transition_probabilities = transition_model(corpus, page, damping_factor)
    for sample in range(n):
        prob = list(transition_probabilities.values())
        page = nprand.choice(all_pages, p=prob, size=1)[0]
        estimated_page_rank[page] += 1
        transition_probabilities = transition_model(corpus, page, damping_factor)
    for page_rank in estimated_page_rank:
        estimated_page_rank[page_rank] /= n
    return estimated_page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    corpus = dict(sorted(corpus.items()))
    estimated_page_rank = {}
    total_pages_corpus = len(corpus)
    for page in corpus:
        estimated_page_rank[page] = 1 / total_pages_corpus
    cpt = 0
    while True:
        should_stop = True
        for page in estimated_page_rank:
            sum_pages = 0
            for i in corpus:
                if i == page or page not in corpus[i]:
                    continue
                num_links = len(corpus[i])
                if num_links == 0:
                    num_links = total_pages_corpus
                sum_pages += estimated_page_rank[i] / num_links
            p = ((1 - damping_factor) / total_pages_corpus +
                                    damping_factor * sum_pages)
            if abs(estimated_page_rank[page] - p) > 0.001:
                should_stop = False
            estimated_page_rank[page] = p
        print(cpt)
        cpt +=1
        if should_stop:
            break
    return estimated_page_rank


if __name__ == "__main__":
    main()
