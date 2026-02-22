use burn::{
    prelude::Backend,
    train::{ClassificationOutput, InferenceStep},
};

use crate::{DemoTransformer, TinyStoryBatch};

impl<B: Backend> InferenceStep for DemoTransformer<B> {
    type Input = TinyStoryBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_classification(item.tokens)
    }
}
