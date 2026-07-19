from typing import TypedDict


class ProductReview(TypedDict):
    product_name: str
    rating: int
    review: str


new_reveiw: ProductReview = {
    'product_name': "Test",
    'rating': 5,
    'review': "Best"
}

print(new_reveiw)
