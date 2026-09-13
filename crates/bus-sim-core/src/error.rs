//! Kernel errors surfaced to Python as `ValueError`.

use std::fmt;

#[derive(Debug)]
pub enum KernelError {
    InvalidConfig(String),
    InvalidScenario(String),
    InvalidAction(i64),
    Conservation(String),
    State(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::InvalidConfig(message) => write!(f, "invalid config: {message}"),
            KernelError::InvalidScenario(message) => write!(f, "invalid scenario: {message}"),
            KernelError::InvalidAction(index) => write!(f, "invalid action index: {index}"),
            KernelError::Conservation(message) => write!(f, "conservation: {message}"),
            KernelError::State(message) => write!(f, "invalid state: {message}"),
        }
    }
}

impl std::error::Error for KernelError {}
