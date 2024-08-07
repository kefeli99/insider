# Insider Trial Day

## Personalized Recommendation Model & Inference

## Installation

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Build Docker File

```bash
docker build -t insider .
```

### Run Container

```
docker run -p 8000:8000 insider
```

### API Docs

API Docs are available at `http://localhost:8000/docs`

## LightFM

1. Hybrid Approach:
   LightFM is a hybrid model that combines collaborative filtering with content-based filtering.

   - It can leverage both user-item interactions and item features.
   - This hybrid approach helps mitigate the cold-start problem for new items or users, as it can make recommendations based on item features even when there's no interaction data.

2. Scalability:
   LightFM is built to handle large-scale datasets efficiently. It uses stochastic gradient descent for optimization, which allows it to scale to large numbers of users and items.

3. Handling Sparse Data:
   E-commerce datasets are often very sparse (users interact with only a tiny fraction of available items). LightFM is designed to handle this sparsity well.

4. Feature Incorporation:
   The ability to incorporate item features (in our case, categories) allows the model to understand similarities between items beyond just user interactions. This can lead to more diverse recommendations.

#### Alternatives Considered:

- Simple collaborative filtering methods (like user-user or item-item similarity) were deemed too simplistic for the complexity of e-commerce data.
- Matrix Factorization techniques (like SVD) don't easily incorporate item features.
- Deep learning models (like neural collaborative filtering) were considered potentially overkill for the dataset size and would require more computational resources.

#### Potential Limitations:

- If the dataset is extremely large, even LightFM might struggle, and we might need to consider distributed computing solutions.
- If we need real-time updates to the model, we might need to consider online learning approaches, which LightFM doesn't natively support.
