Title: Python: Mocking for tests
Date: 26-05-2016
Category: tutorials

Software often relies on third party services to function. Third party services
can be anything, from a random number generator to a REST API. We'll take a
simple HTTP API as an example. When designing unittests for the parts of your
code that interact with the API, you need some way to `mock` out the
interaction with the API with something predictable that's not dependent on the
API, or the internet, or anything that you can't reliably predict; if you
don't, your unittests will occasionally fail and that is not what you want to
happen.

For mocking objects we can use `unittest.mock` in Python 3 (there's a backport
available for Python 2 on PyPI called `Mock`). Let's specify our example
problem example first:

    :::python
    # filename: app.py

    from requests import post


    def call_api(data):
        if not 'username' in data:
            raise KeyError('username is not in data')
        headers = {'Content-Type': 'application/json'}
        return post('http://someurl.com', json=data, headers=headers).json()


    def get_user_data(username):
        data = {'username': username}
        response = call_api(data)
        return response

We can use this as a very simple case study. `get_user_data` gets passed in a
username as string and calls the API with the username packed into a
dictionary.  `call_api` checks if the data dictionary contains a key
`'username'` and if so returns the API response for the request.

Now if we were to write unittests for this as is, we would be dependent on the
`requests.post` function accessing the internet and return a response that we
could maybe verify. There's a lot of potential problems with this (changing data
on server-side being a big one), so we will patch out the API request with
something we can predict.

    :::python
    # filename: test.py

    from unittest.mock import patch
    import app
    import unittest


    class TestApp(unittest.TestCase):

        @patch('app.call_api', side_effect=KeyError)
        def test_get_user_data_error(self, mock_call_api):
            self.assertRaises(KeyError, app.get_user_data, 'user')

        @patch('app.call_api', side_effect=lambda x: x)
        def test_get_user_data_success(self, mock_call_api):
            self.assertEquals(app.get_user_data('user1'), 'user1')

        @patch('app.call_api', return_value=3)
        def test_get_user_data_success2(self, mock_call_api):
            self.assertEquals(app.get_user_data('username'), 3)


    if __name__ == '__main__':
        unittest.main()

When I'm working with `unittest.mock`, I primarily use the patch decorator, so
that's what I decided to show here. It takes a function to patch and some
optional keyword arguments, such as `side_effect` or `return_value` (there's
more, see the documentation!). It then passes the patched function as an
argument to the function it decorates. In this example, we don't actually do
anything with the patched function directly, but it can be useful in other
situations.

So what is happening? The first test case tests that the `call_api` function
throws a `KeyError`. We accomplish this by patching the function and telling
it to throw a `KeyError` when it's called through the `side_effect` argument.
The second test case uses the identity function `lambda x: x` as a side effect;
essentially this means, *return whatever you were passed*. We supply one
argument to the function, so it will return that same argument.
In the third test case, we simply set the return value of `call_api` to 3 and
verify that `get_user_data` indeed returns 3.

Simple stuff! I will probably write another post that goes a little more in
depth on this subject as I become more comfortable with it. You can accomplish
some pretty cool stuff with the `Mock` and `MagicMock` classes!
