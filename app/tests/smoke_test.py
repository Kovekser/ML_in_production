def test_smoke(app, client):
    response = client.get(app.url_path_for("default"))
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
