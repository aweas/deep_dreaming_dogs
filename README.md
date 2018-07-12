# deep_dreaming_dogs
Own implementation of deep dreaming algorithm from scratch, CNN training included. All files are in jupyter notebooks,
but I will release them as .py for readability after finishing the project.

File network.py also includes example of loading big dataset using new Dataset and Iterator tensorflow API

Training set used: https://www.kaggle.com/c/dogs-vs-cats

### Sample results
## Inputs
<div class="row">
  <div class="column">
    <img src="https://i.imgur.com/y5kr7ve.jpg" width="256" height="256"/>
  </div>
  <div class="column">
    <img src="https://i.imgur.com/e7gLp6d.jpg" width="256" height="256"/>
  </div>
  <div class="column">
    <img src="https://i.imgur.com/R0viaB9.jpg" width="256" height="256" />
  </div>
</div> 
<img src="https://i.imgur.com/ITyp5XK.jpg" width="256" height="256" />

## Results
![Processed dog photo](https://i.imgur.com/8CNtpeQ.jpg)
![Processed cat photo](https://i.imgur.com/5Jd6fk9.jpg)
![Cat photo 2](https://i.imgur.com/d35Az8q.jpg)
![Dog photo 2](https://i.imgur.com/bvgqRk6.jpg)

## Results from dreaming over random samples
![Random dreaming](https://i.imgur.com/zr8ZyXa.png)

## Some more results from dreaming over photos
![Dog](https://i.imgur.com/XuQ9cdb.png)
![Cat](https://i.imgur.com/8ucaLnd.png)
