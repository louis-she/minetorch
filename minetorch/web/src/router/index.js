import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      component: () => import('components/_common/container'),
      redirect: '/list',
      children: [
        {
          path: '/list',
          name: 'List',
          component: () => import('pages/list')
        }
      ]
    },
    {
      path: '*',
      redirect: '/list'
    }
  ]
})
