import pytest
from scrapegraphai.utils.read_mode import transform_link

def test_transform_link():
    original_link = 'https://android.stackexchange.com/questions/218970'
    expected_hash = '07036109224c60335e35e3b4c22dd02cf775f69d4430245c4c454aff570d6787'
    expected_output = f'chrome-distiller://00000000-0000-0000-0000-000000000000_{expected_hash}/?url=https%3A//android.stackexchange.com/questions/218970'
    
    transformed_link = transform_link(original_link)
    
    assert transformed_link == expected_output, f"Expected {expected_output}, but got {transformed_link}"

if __name__ == "__main__":
    pytest.main()
