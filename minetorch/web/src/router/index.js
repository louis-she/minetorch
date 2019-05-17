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
        },
        {
          path: '/experiment/:experimentId/edit',
          name: 'EditExperiment',
          component: () => import('pages/experiment/edit'),
          children: [
            {
              path: 'summary',
              name: 'EditExperimentSummary',
              component: () => import('pages/experiment/edit/summary')
            },
            {
              path: ':componentName',
              name: 'EditExperimentComponent',
              component: () => import('pages/experiment/edit/component')
            }
          ]
        }
      ]
    },
    {
      path: '*',
      redirect: '/list'
    }
  ]
})
